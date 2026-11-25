"""Build a self-contained daily report from the current research snapshot."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWERS = ("Claude Opus 5.5", "GPT 6.2 Sol")
HOSTS = {"bazzite": "bazzite", "macbook_pro": "MacBook Pro", "macbook_air": "MacBook Air"}

CSS = """
* { box-sizing: border-box }
body { margin: 0; color: var(--foreground); font: 15px/1.45 var(--font-sans), sans-serif }
main { width: 100%; overflow-wrap: anywhere }
section + section { margin-top: 24px; border-top: 1px solid var(--border); padding-top: 16px }
h1, h2, h3, p, dl, dd, ol, ul, figure { margin: 0 }
h1 { font-size: 1.05rem } h2 { font-size: 1.16rem; margin-bottom: 10px }
h3 { font-size: .95rem; margin-bottom: 6px }
.meta, .pills { display: flex; align-items: baseline; gap: 5px 12px; flex-wrap: wrap }
.meta { margin-bottom: 10px } .meta time { margin-left: auto }
time, .note, dt, figcaption { color: var(--muted-foreground); font-size: .82rem }
.kpis, .board { display: grid; gap: 8px; grid-template-columns: repeat(4, minmax(0, 1fr)) }
.kpis > div { background: var(--card); padding: 16px; border-radius: var(--radius) }
.kpis dd { font-size: 1.3rem; font-weight: 650; font-variant-numeric: tabular-nums }
.board { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px }
.done { color: var(--success) } .wip { color: var(--info) } .risk { color: var(--destructive) }
ul, ol { padding-inline-start: 1.25rem } li + li { margin-top: .3rem }
.host { margin-bottom: 8px } .pill { border: 1px solid var(--border); border-radius: 1em; padding: 1px 8px; font-size: .78rem }
details, figure + figure, .subhead { margin-top: 12px } summary { cursor: pointer }
svg { display: block; width: 100%; color: var(--muted-foreground) }
svg text { fill: currentColor; font: 12px var(--font-sans), sans-serif }
.axis { fill: none; stroke: var(--border) } .front { stroke: var(--primary); fill: none; stroke-dasharray: 4 3 }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .85rem }
th, td { padding: 4px 0; text-align: right; border-bottom: 1px solid var(--border) }
th:first-child, td:first-child { text-align: left }
@media (max-width: 580px) {
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)) }
  .board { grid-template-columns: 1fr } .meta time { margin-left: 0 }
}
"""


def safe(value):
    return escape(str(value), quote=True)


def item_text(item):
    if isinstance(item, dict):
        return next((str(item[k]) for k in ("summary", "title", "task", "name", "description") if item.get(k)), "")
    return str(item)


def listing(items, empty, tag="ul"):
    labels = [item_text(item) for item in items]
    entries = "".join(f"<li>{safe(label)}</li>" for label in labels if label)
    return f"<{tag}>{entries}</{tag}>" if entries else f'<p class="note">{empty}</p>'


def number(row, *keys):
    for key in keys:
        value = row.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            return value
    return None


def kpi_strip(kpis):
    speedups = [(name, number(kpis, key)) for name, key in (
        ("PE", "best_speedup_vs_pe"), ("PS", "best_speedup_vs_ps"))]
    # The two maxima may come from different candidates. Keep their baselines explicit.
    speed = " · ".join(f"{v:.3g}× {name}" for name, v in speedups if v is not None)
    tests = f'{kpis.get("tests_passing", 0)} tests · {kpis.get("audits_completed", 0)} audits'
    metrics = (("Experiments Run", kpis.get("experiments_run", 0)),
               ("Verified Wins", kpis.get("verified_wins", 0)),
               ("Best Speedup vs SOTA", speed or "Pending"), ("Tests & Dual Audits", tests))
    return '<dl class="kpis">' + "".join(
        f"<div><dt>{safe(k)}</dt><dd>{safe(v)}</dd></div>" for k, v in metrics) + "</dl>"


def scatter(group, points):
    xmax = max(x for _, x, _ in points) or 1
    ymax = max(y for _, _, y in points) or 1

    def xy(x, y):
        return f"{54 + 252 * (x / xmax):.2f},{174 - 146 * (y / ymax):.2f}"

    frontier = []
    best_error = math.inf
    for x, y in sorted(set((x, y) for _, x, y in points)):
        if y < best_error:
            frontier.append(xy(x, y))
            best_error = y
    svg = ['<svg viewBox="0 0 330 214" width="100%" height="214" role="img" aria-label="Latency versus relative polar error">',
           '<path class="axis" d="M54 28V174H306"/>',
           '<text x="4" y="16">Relative polar error</text>',
           f'<text x="50" y="32" text-anchor="end">{ymax:.2g}</text>',
           '<text x="50" y="178" text-anchor="end">0</text><text x="54" y="191">0</text>',
           f'<text x="306" y="191" text-anchor="end">{xmax:.3g}</text>',
           '<text x="180" y="209" text-anchor="middle">Latency (ms)</text>',
           f'<polyline class="front" points="{" ".join(frontier)}"/>']
    legend = []
    for i, (name, x, y) in enumerate(points):
        color = f'var(--chart-{1 + i % 6})'
        cx, cy = xy(x, y).split(",")
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{color}"><title>{safe(name)}: {x:.4g} ms, error {y:.4g}</title></circle>')
        legend.append(f'<tr><td><span aria-hidden="true" style="color:{color}">●</span> {safe(name)}</td><td>{x:.4g}</td><td>{y:.4g}</td></tr>')
    return (f'<figure><h3>{safe(group)}</h3>{"".join(svg)}</svg>'
            '<figcaption>Lower is better on both axes. The dashed line joins the Pareto frontier.</figcaption>'
            '<table><thead><tr><th>Method</th><th>Latency (ms)</th><th>Relative error</th></tr></thead>'
            f'<tbody>{"".join(legend)}</tbody></table></figure>')


def measurements(rows):
    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        latency = number(row, "latency_ms", "median_ms", "time_ms", "median_case_latency_ms")
        error = number(row, "polar_error", "relative_polar_error", "polar_error_rel", "rel_polar_error",
                       "relative_frobenius_error", "median_relative_frobenius_error")
        if latency is None or error is None:
            continue
        shape = row.get("shape", "")
        if isinstance(shape, (list, tuple)):
            shape = " × ".join(map(str, shape))
        group = " · ".join(str(v) for v in (row.get("backend"), row.get("dtype"), row.get("case"), shape) if v)
        groups[group or "Benchmark measurements"].append((row.get("method", row.get("name", "Unnamed method")), latency, error))
    return groups


def benchmarks(root, state):
    path = root / "results" / f'cycle{state.get("cycle", 1)}_head_to_head.json'
    groups = measurements(state.get("leaderboard", []))
    source = "Research state leaderboard"
    excluded = []
    if not groups and path.is_file():
        data = json.loads(path.read_text())
        source = path.name
        if isinstance(data, dict) and "all" in data.get("summary", {}):
            # Tier 1 records separate medians over the same battery of cases.
            # Missing and divergent results cannot compete on the frontier.
            rows = []
            for method, metrics in data["summary"]["all"].items():
                count = metrics["cases"]
                if count != metrics["timed_results"] or count != metrics["numeric_results"] or metrics["divergence_rate"]:
                    excluded.append(method)
                    continue
                rows.append(dict(metrics, method=method, backend=data.get("device"),
                                 case=f"Median across {count} cases"))
            groups = measurements(rows)
        else:
            rows = data if isinstance(data, list) else data.get("rows", data.get("results", data.get("leaderboard", [])))
            groups = measurements(rows)
    plots = "".join(scatter(group, points) for group, points in groups.items())
    result = (f'<p class="note">{safe(source)}</p>{plots}' if plots else
              '<p class="note">Awaiting paired latency and polar error measurements.</p>')
    if excluded:
        result += '<p class="note">Incomplete or divergent runs: ' + safe(", ".join(excluded)) + '.</p>'
    return result


def benchmark_records(root, cycle):
    """Summarize the optional device runs without promoting them to verified wins."""
    entries = []
    for path in sorted((root / "results").glob(f"*_cycle{cycle}.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("results"), list) or "backend" not in data:
            continue
        rows = data["results"]
        finite = sum(isinstance(row, dict) and row.get("finite") is True for row in rows)
        mode = "quick" if data.get("quick") else "full"
        host = data.get("hardware", {}).get("hostname", data["backend"])
        entries.append(f'{host} · {data["backend"]} · {mode} · {finite}/{len(rows)} finite outputs ({path.name})')
    if not entries:
        return ""
    return '<details><summary>Device benchmark records</summary>' + listing(entries, "") + "</details>"


def fleet_audits(state, ledger):
    hosts = []
    for key, label in HOSTS.items():
        host = state.get("fleet", {}).get(key, {})
        status = safe(host.get("status", "unknown")).replace("_", " ")
        hosts.append(f'<div class="host"><div class="pills"><strong>{label}</strong><span class="pill">{status}</span></div>'
                     f'<p class="note">{safe(host.get("hardware", ""))}</p></div>')
    audits = state.get("audits", [])
    if isinstance(audits, dict):
        audits = [dict(value, model=name) if isinstance(value, dict) else {"model": name, "verdict": value}
                  for name, value in audits.items()]
    entries = []
    for model in REVIEWERS:
        verdicts = [str(a.get("verdict") or a.get("status") or "Pending") for a in audits
                    if isinstance(a, dict) and model.casefold() == str(a.get("model") or a.get("reviewer")).casefold()]
        entries.append(f"<li><strong>{model}</strong> · {safe(' / '.join(verdicts) or 'Pending')}</li>")
    return ("".join(hosts) + '<h3 class="subhead">Adversarial audit verdicts</h3><ul>' + "".join(entries) +
            '</ul><details><summary>Recent research ledger entries</summary>' +
            listing(ledger[-3:][::-1], "No entries recorded.") + '</details>')


def generate(repo_root, report_date, output=None):
    root = Path(repo_root)
    state = json.loads((root / "state/RESEARCH_STATE.json").read_text())
    ledger = [json.loads(line) for line in (root / "state/ledger.jsonl").read_text().splitlines() if line.strip()]
    stamp = safe(state.get("last_updated", report_date))
    board = []
    for key, label, style, empty in (
        ("completed", "Completed", "done", "No completions recorded."),
        ("wip", "Work In Progress", "wip", "No active work recorded."),
        ("blockers", "Blockers & Risks", "risk", "No blockers recorded."),
    ):
        board.append(f'<div><h3 class="{style}">{safe(label)}</h3>{listing(state.get(key, []), empty)}</div>')
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>domainus/ai daily research · {safe(report_date)}</title><style>{CSS}</style></head>
<body><main>
<section aria-label="Research overview"><header class="meta"><h1>domainus/ai</h1><strong>Cycle {safe(state.get("cycle", "—"))}</strong>
<time datetime="{stamp}">{stamp}</time></header>{kpi_strip(state.get("kpis", {}))}</section>
<section><h2>Benchmark Comparison &amp; Pareto Chart</h2>{benchmarks(root, state)}</section>
<section><h2>Status Board</h2><div class="board">{"".join(board)}</div></section>
<section><h2>Fleet Hardware &amp; Multi-Model Adversarial Audit Ledger</h2>{fleet_audits(state, ledger)}{benchmark_records(root, state.get("cycle", 1))}</section>
<section><h2>Next Autonomous Sprint Queue</h2>{listing(state.get("next_queue", []), "No tasks queued.", "ol")}</section>
</main></body></html>
"""
    output = Path(output) if output else root / "reports/daily" / f"{report_date}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(generate(args.repo_root, args.date, args.output))
