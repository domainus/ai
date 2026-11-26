"""Build disjoint GPT-2-tokenized train/val/test shards from real English pretraining text.

The script streams ``HuggingFaceFW/fineweb-edu`` (``sample-10BT``) first. If the stream fails,
or its projected rate misses ``--max-seconds``, it starts again from ``Salesforce/wikitext``
(``wikitext-103-raw-v1``). Sources are never mixed in one build.

Each document is routed to exactly one split by ``sha1(text) % 15`` (0 -> val, 1 -> test,
others -> train), after exact-text deduplication. A split stops taking documents once it reaches
its token target, so the splits are disjoint at document level. Every document is prefixed with
the GPT-2 end-of-text token 50256.

Outputs (``uint16`` raw memmaps, no header) under ``artifacts/real_text/``:
    train.bin, val.bin, test.bin, doc_hashes.json, manifest.json

Run:
    PYTHONPATH=. .venv/bin/python research/polar_opt/real_text_data.py
Verify only:
    PYTHONPATH=. .venv/bin/python research/polar_opt/real_text_data.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "artifacts" / "real_text"
ENCODING = "gpt2"
VOCAB_SIZE = 50257
PADDED_VOCAB_SIZE = 50304  # next multiple of 64, for tensor cores
EOT = 50256
SPLITS = ("train", "val", "test")
DEFAULT_TARGETS = {"train": 26_000_000, "val": 2_000_000, "test": 2_000_000}
ROUTING_MODULUS = 15
# The fleet rule fixes the User-Agent for every web request this project makes.
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"


class TooSlow(RuntimeError):
    """The source cannot fill the targets within the time limit."""


def route(digest: str) -> str:
    bucket = int(digest[:8], 16) % ROUTING_MODULUS
    return "val" if bucket == 0 else "test" if bucket == 1 else "train"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _encoder():
    import tiktoken

    enc = tiktoken.get_encoding(ENCODING)
    assert enc.n_vocab == VOCAB_SIZE and enc.eot_token == EOT
    return enc


def build_shards(documents: Iterable[str], out_dir: Path, *, targets: dict[str, int] | None = None,
                 source: dict, max_seconds: float | None = None, probe_seconds: float = 90.0,
                 batch_documents: int = 512, encoder=None) -> dict:
    """Tokenize ``documents`` into disjoint split shards and write the manifest.

    ``max_seconds`` limits the build. After ``probe_seconds`` the projected finish time is
    checked, and :class:`TooSlow` is raised early when it exceeds the limit. ``encoder`` must
    provide ``encode_ordinary_batch``; the default is tiktoken's GPT-2 encoding.
    """
    targets = dict(targets or DEFAULT_TARGETS)
    enc = encoder or _encoder()
    tokens = {name: [] for name in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    hashes = {name: [] for name in SPLITS}
    seen: set[str] = set()
    stats = {"documents_seen": 0, "duplicates_skipped": 0, "empty_skipped": 0, "overflow_skipped": 0}
    total_target = sum(targets.values())
    start = time.perf_counter()

    def full() -> bool:
        return all(counts[name] >= targets[name] for name in SPLITS)

    def flush(batch: list[tuple[str, str]]) -> None:
        encoded = enc.encode_ordinary_batch([text for _, text in batch], num_threads=os.cpu_count() or 4)
        for (digest, _), ids in zip(batch, encoded):
            split = route(digest)
            if counts[split] >= targets[split]:
                stats["overflow_skipped"] += 1
                continue
            array = np.empty(len(ids) + 1, dtype=np.uint16)
            array[0] = EOT
            array[1:] = ids
            tokens[split].append(array)
            counts[split] += len(array)
            hashes[split].append(digest)

    batch: list[tuple[str, str]] = []
    for text in documents:
        stats["documents_seen"] += 1
        if not text or not text.strip():
            stats["empty_skipped"] += 1
            continue
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen:
            stats["duplicates_skipped"] += 1
            continue
        seen.add(digest)
        batch.append((digest, text))
        if len(batch) >= batch_documents:
            flush(batch)
            batch = []
            if full():
                break
            elapsed = time.perf_counter() - start
            done = sum(min(counts[n], targets[n]) for n in SPLITS) / total_target
            if max_seconds is not None and elapsed > probe_seconds and done > 0:
                if elapsed / done > max_seconds:
                    raise TooSlow(f"projected {elapsed / done:.0f}s > limit {max_seconds:.0f}s "
                                  f"({done:.1%} after {elapsed:.0f}s)")
    if batch:
        flush(batch)
    short = {n: (counts[n], targets[n]) for n in SPLITS if counts[n] < targets[n]}
    if short:
        raise RuntimeError(f"Source exhausted before reaching targets: {short}")

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in SPLITS:
        data = np.concatenate(tokens[name])
        assert int(data.max()) < VOCAB_SIZE
        tmp = out_dir / f"{name}.bin.tmp"
        data.tofile(tmp)
        tmp.replace(out_dir / f"{name}.bin")
        splits[name] = {
            "file": f"{name}.bin", "tokens": int(data.size), "documents": len(hashes[name]),
            "target_tokens": targets[name], "sha256": _sha256(out_dir / f"{name}.bin"),
            "doc_hash_digest": hashlib.sha256("\n".join(sorted(hashes[name])).encode()).hexdigest(),
        }
    (out_dir / "doc_hashes.json").write_text(json.dumps(hashes) + "\n")
    manifest = {
        "source": source, "encoding": ENCODING, "vocab_size": VOCAB_SIZE,
        "padded_vocab_size": PADDED_VOCAB_SIZE, "eot_token": EOT, "dtype": "uint16",
        "layout": "raw little-endian uint16 memmap, no header; each document is EOT + tokens",
        "routing": f"sha1(utf8 text)[:8] % {ROUTING_MODULUS}: 0 -> val, 1 -> test, else -> train; "
                   "exact-text dedup; a split stops taking documents at its target",
        "splits": splits, **stats,
        "build_seconds": round(time.perf_counter() - start, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_manifest(directory: Path = DEFAULT_DIR) -> dict:
    """Check sizes, SHA-256 checksums, token range and document disjointness. Raise on mismatch."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["encoding"] != ENCODING or manifest["dtype"] != "uint16":
        raise ValueError("Unexpected encoding or dtype in manifest")
    hashes = json.loads((directory / "doc_hashes.json").read_text())
    for name in SPLITS:
        entry = manifest["splits"][name]
        path = directory / entry["file"]
        if path.stat().st_size != 2 * entry["tokens"]:
            raise ValueError(f"{name}: size {path.stat().st_size} != 2 * {entry['tokens']} tokens")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"{name}: SHA-256 mismatch")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        if int(data.max()) >= VOCAB_SIZE or data[0] != EOT:
            raise ValueError(f"{name}: token out of range or missing leading EOT")
        if len(hashes[name]) != entry["documents"] or int((data == EOT).sum()) != entry["documents"]:
            raise ValueError(f"{name}: document count mismatch")
        digest = hashlib.sha256("\n".join(sorted(hashes[name])).encode()).hexdigest()
        if digest != entry["doc_hash_digest"]:
            raise ValueError(f"{name}: document hash list does not match the manifest")
    sets = {name: set(hashes[name]) for name in SPLITS}
    for i, a in enumerate(SPLITS):
        if len(sets[a]) != len(hashes[a]):
            raise ValueError(f"{a}: duplicate documents")
        for b in SPLITS[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f"{a} and {b} share {len(sets[a] & sets[b])} documents")
    return manifest


def _set_user_agent() -> None:
    """Send the fixed User-Agent on Hugging Face Hub requests."""
    from huggingface_hub.utils import _headers

    _headers._http_user_agent = lambda *args, **kwargs: USER_AGENT


def fineweb_documents() -> Iterator[str]:
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in stream:
        yield row["text"]


def wikitext_documents() -> Iterator[str]:
    """Yield whole WikiText-103 articles (official train, validation and test splits)."""
    from datasets import load_dataset

    for split in ("train", "validation", "test"):
        article: list[str] = []
        for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split):
            line = row["text"]
            # Article titles are " = Title = "; section headings have two or more "=".
            if line.startswith(" = ") and not line.startswith(" = = ") and article:
                yield "".join(article)
                article = []
            article.append(line)
        if article:
            yield "".join(article)


SOURCES = {
    "fineweb-edu": (fineweb_documents, {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
                                        "split": "train", "streaming": True}),
    "wikitext-103": (wikitext_documents, {"dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1",
                                          "split": "train+validation+test (articles)", "streaming": False}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--source", choices=["auto", *SOURCES], default="auto")
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TARGETS["train"])
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_TARGETS["val"])
    parser.add_argument("--test-tokens", type=int, default=DEFAULT_TARGETS["test"])
    parser.add_argument("--max-seconds", type=float, default=1800.0,
                        help="FineWeb-Edu time limit before falling back to WikiText-103")
    parser.add_argument("--verify", action="store_true", help="only verify an existing build")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_manifest(args.out)["splits"], indent=2))
        return
    _set_user_agent()
    targets = {"train": args.train_tokens, "val": args.val_tokens, "test": args.test_tokens}
    order = list(SOURCES) if args.source == "auto" else [args.source]
    failures = []
    for name in order:
        documents, source = SOURCES[name]
        source = {**source, "name": name, "fallback_from": failures or None}
        print(f"Building from {name} ...", flush=True)
        try:
            manifest = build_shards(documents(), args.out, targets=targets, source=source,
                                    max_seconds=args.max_seconds if name == "fineweb-edu" else None)
        except Exception as error:  # noqa: BLE001 - any source failure triggers the fallback
            print(f"  {name} failed: {type(error).__name__}: {error}", flush=True)
            failures.append({"source": name, "error": f"{type(error).__name__}: {error}"})
            continue
        verify_manifest(args.out)
        print(json.dumps({k: manifest[k] for k in ("source", "splits", "build_seconds")}, indent=2))
        return
    raise SystemExit(f"All sources failed: {failures}")


if __name__ == "__main__":
    main()
