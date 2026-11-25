import json
import re
import subprocess
import sys
from html.parser import HTMLParser

import pytest

from reports.generate_infographic import ROOT


class ReportParser(HTMLParser):
    """Check balanced markup and collect the parts relevant to the T3 frame."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.tags = []
        self.text = []
        self.css = []
        self.charts = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag not in ("meta", "br", "hr"):
            self.stack.append(tag)
        if tag == "svg":
            self.charts.append(dict(attrs))

    def handle_endtag(self, tag):
        assert self.stack.pop() == tag

    def handle_data(self, data):
        if self.stack and self.stack[-1] == "style":
            self.css.append(data)
        else:
            self.text.append(data)


def verify(path):
    content = path.read_text()
    assert content.lower().startswith("<!doctype html>")
    doc = ReportParser()
    doc.feed(content)
    doc.close()
    assert not doc.stack
    assert doc.tags.count("section") == 5
    assert doc.tags.count("dt") == 4
    assert {"script", "iframe", "link", "img"}.isdisjoint(doc.tags)
    text = "".join(doc.text)
    for label in ("domainus/ai", "Experiments Run", "Verified Wins", "Best Speedup vs SOTA",
                  "Tests & 4-Model Audits", "Benchmark Comparison & Pareto Chart", "Status Board",
                  "Fleet Hardware & Multi-Model Adversarial Audit Ledger", "Next Autonomous Sprint Queue",
                  "Completed", "Work In Progress", "Blockers & Risks", "Claude Opus 5.5",
                  "Claude Fable 5.2", "GPT 6.2 Astra", "Gemini 4 Neon",
                  "bazzite", "MacBook Pro", "MacBook Air"):
        assert label in text
    css = "".join(doc.css).lower()
    assert not re.search(r"(?:\d|\.)vh\b", content, re.IGNORECASE)
    assert not re.search(r"#[0-9a-f]{3,8}\b", css)
    for selectors, props in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if any(s.strip() in ("html", "body", "main", ":root", "*") for s in selectors.split(",")):
            assert not re.search(r"\b(?:background(?:-color)?|border|padding(?:-inline)?)\s*:", props)
            assert not re.search(r"\bheight\s*:\s*100\s*%", props)
    for chart in doc.charts:
        assert chart["height"].isdigit() and chart["viewbox"] and chart["width"] == "100%"
    return doc, text


def test_live_state_cli(tmp_path):
    output = tmp_path / "custom/report.html"
    subprocess.run([sys.executable, str(ROOT / "reports/generate_infographic.py"),
                    "--output", str(output)], cwd=tmp_path, check=True)
    doc, text = verify(output)
    state = json.loads((ROOT / "state/RESEARCH_STATE.json").read_text())
    assert f'Cycle {state["cycle"]}' in text and state["last_updated"] in text
    assert len(doc.charts) >= 1
    assert "Awaiting paired" not in text
    h2h = ROOT / f'results/cycle{state["cycle"]}_head_to_head.json'
    if h2h.is_file():
        assert "polar_express_fp8x2_bf16" in text


@pytest.mark.parametrize("source", ["leaderboard", "results", "tier1", "empty"])
def test_report_from_snapshot(tmp_path, source):
    state = {
        "cycle": 1, "last_updated": "2026-11-25T07:00:00Z",
        "audits": [{"model": "Claude Opus 5.5", "verdict": "Conditional pass"},
                   {"auditor": "Claude Fable 5.2 (high)", "verdict": "Pass after fix"},
                   {"reviewer": "GPT 6.2 Astra", "verdict": "Recheck variance"},
                   {"auditor": "Gemini 4 Neon (high)", "verdict": "Pass with caveats"}],
        "next_queue": ["Repeat slow case"], "blockers": ["<script>untrusted</script>"],
        "kpis": {"experiments_run": 5, "best_speedup_vs_pe": 1.25},
    }
    rows = [{"method": "First & fast", "shape": [16, 64], "latency_ms": 3, "polar_error": .2},
            {"method": "Second", "shape": [16, 64], "latency_ms": 5, "polar_error": .1}]
    if source == "leaderboard":
        state["leaderboard"] = rows
    elif source == "results":
        (tmp_path / "results").mkdir()
        (tmp_path / "results/cycle1_head_to_head.json").write_text(json.dumps({"results": rows}))
    elif source == "tier1":
        (tmp_path / "results").mkdir()
        metrics = {"cases": 9, "numeric_results": 9, "timed_results": 9, "divergence_rate": 0}
        methods = {r["method"]: dict(metrics, median_case_latency_ms=r["latency_ms"],
                                     median_relative_frobenius_error=r["polar_error"]) for r in rows}
        methods["Unstable"] = dict(metrics, divergence_rate=1)
        (tmp_path / "results/cycle1_head_to_head.json").write_text(json.dumps({"summary": {"all": methods}}))
    (tmp_path / "state").mkdir()
    (tmp_path / "state/RESEARCH_STATE.json").write_text(json.dumps(state))
    (tmp_path / "state/ledger.jsonl").write_text('{"summary":"Fleet initialized"}\n')
    subprocess.run([sys.executable, str(ROOT / "reports/generate_infographic.py"), "--repo-root",
                    str(tmp_path), "--date", "2026-11-25"], check=True)
    doc, text = verify(tmp_path / "reports/daily/2026-11-25.html")
    assert "Conditional pass" in text and "Pass after fix" in text
    assert "Recheck variance" in text and "Pass with caveats" in text
    assert "1.25× PE" in text and "Fleet initialized" in text
    assert "<script>untrusted</script>" in text
    if source == "empty":
        assert not doc.charts and "Awaiting paired" in text
    else:
        assert len(doc.charts) == 1 and doc.tags.count("circle") == 2
        assert "First & fast" in text and "Second" in text
        if source == "tier1":
            assert "Median across 9 cases" in text and "Incomplete or divergent runs: Unstable" in text
        else:
            assert "16 × 64" in text
