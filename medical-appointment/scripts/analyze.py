"""Deep-dive analysis of a local evaluation run.

Requires a running server (`python api.py`, or run scripts/run_local_eval.py
first to also get the official-format sanity numbers archived). Produces:

    reports/analyze_<timestamp>_<git-sha>.md     -- read this one. It's built
        to be pasted whole into an LLM chat to get a diagnosis of what's
        failing and why, and what to try next.
    reports/analyze_<timestamp>_<git-sha>.json   -- the same data, structured,
        for your own scripting.
    reports/analyze_<timestamp>_<git-sha>_plots.png  -- tIoU distribution,
        per-type accuracy, latency histogram (skipped if matplotlib isn't
        installed; the .md/.json still get written).

Usage:
    python scripts/analyze.py
    python scripts/analyze.py --worst-n 15
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import local_evaluator as official  # noqa: E402
from solution.eval_support import DetailRow, replay_with_detail  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze")


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "nogit"


def _fmt_span(span: Optional[tuple]) -> str:
    if span is None:
        return "-"
    start, end = span
    return f"{start:.2f}-{end:.2f}s"


def _worst_missed_positives(details: List[DetailRow], n: int) -> List[DetailRow]:
    """True yes, answered no or unanswered -- wrong on both halves of the score."""
    return [d for d in details if d.label == official.YES and d.prediction != official.YES][:n]


def _worst_hard_negative_false_positives(details: List[DetailRow], n: int) -> List[DetailRow]:
    """The topical-similarity trap the README warns about by name."""
    return [
        d for d in details if d.question_type == "hard_negative" and d.prediction == official.YES
    ][:n]


def _worst_tiou_true_positives(details: List[DetailRow], n: int) -> List[DetailRow]:
    """Answered right, but the span is off -- localizer needs work, not the verifier."""
    rows = [d for d in details if d.tiou is not None and d.prediction == official.YES]
    rows.sort(key=lambda d: d.tiou)
    return rows[:n]


def _markdown_row(d: DetailRow) -> str:
    said = "yes" if d.prediction == official.YES else ("no" if d.prediction == 0 else "unanswered")
    wanted = "yes" if d.label == official.YES else "no"
    tiou = f"{d.tiou:.3f}" if d.tiou is not None else "-"
    return (
        f"| `{d.audio_filename}` | {d.question_type} | {d.question} "
        f"| said {said}, wanted {wanted} "
        f"| gold {_fmt_span(d.gold_span)} / pred {_fmt_span(d.predicted_span)} | {tiou} |"
    )


def _table(rows: List[DetailRow]) -> List[str]:
    header = [
        "| conversation | type | question | result | spans | tIoU |",
        "|---|---|---|---|---|---|",
    ]
    if not rows:
        return header + ["| *(none)* | | | | | |"]
    return header + [_markdown_row(d) for d in rows]


def _span_length_stats(details: List[DetailRow]) -> Optional[dict]:
    """Predicted vs. gold span duration, for true positives with both
    spans present. This is what actually tells you *why* tIoU is low:
    predicted consistently wider than gold means the extraction is too
    loose (the fix is tightening, e.g. solution/verify.py's fine-trim
    pass); predicted consistently narrower means it's clipping real
    evidence; close durations with still-low tIoU means it's a placement
    problem, not a length one, and tightening further won't help.
    """

    pairs = [
        (d.predicted_span[1] - d.predicted_span[0], d.gold_span[1] - d.gold_span[0])
        for d in details
        if d.prediction == official.YES and d.label == official.YES
        and d.predicted_span is not None and d.gold_span is not None
    ]
    if not pairs:
        return None

    pred_durations = sorted(p for p, _ in pairs)
    gold_durations = sorted(g for _, g in pairs)
    n = len(pairs)
    mid = n // 2
    return {
        "n": n,
        "pred_mean": sum(pred_durations) / n,
        "pred_median": pred_durations[mid],
        "gold_mean": sum(gold_durations) / n,
        "gold_median": gold_durations[mid],
    }


def build_report(statistics, details: List[DetailRow], worst_n: int, name: str) -> str:
    lines = [
        f"# Local evaluation analysis -- {name}",
        "",
        "Paste this whole file into an LLM chat (or read it yourself) for a",
        "diagnosis of what's failing and why, and what to try next.",
        "",
        "## Headline",
        "",
        f"- Score: **{statistics.final_score:.3f}** "
        f"(accuracy={statistics.accuracy:.3f}, mean tIoU={statistics.mean_tiou:.3f})",
        f"- Conversations: {statistics.conversations} "
        f"({statistics.failed_conversations} failed requests, {statistics.timeouts} timeouts"
        f"{', ABORTED early' if statistics.aborted else ''})",
        "",
        "## Accuracy by question type",
        "",
        "| type | accuracy | n |",
        "|---|---|---|",
    ]
    for qtype in ("positive", "hard_negative", "off_topic"):
        correct, total = statistics.by_type.get(qtype, [0, 0])
        if total:
            lines.append(f"| {qtype} | {correct / total:.3f} | {total} |")

    lines += [
        "",
        "## Evidence localization",
        "",
        f"- mean tIoU (scored): **{statistics.mean_tiou:.3f}** over "
        f"{len(statistics.tious)} annotated-yes questions",
        f"- no span returned: {statistics.missing_spans}",
        f"- tIoU when answered yes (diagnostic, not scored): "
        f"{statistics.mean_tiou_answered_yes:.3f}, n={len(statistics.tious_answered_yes)}",
        "",
        "A high diagnostic number next to a low scored mean tIoU means the localizer is fine",
        "when it notices, but is missing too many positives outright (fix recall / threshold).",
        "Both low means the localizer itself -- the span, not the yes/no decision -- needs work.",
    ]

    span_stats = _span_length_stats(details)
    if span_stats is not None:
        ratio = span_stats["pred_mean"] / span_stats["gold_mean"] if span_stats["gold_mean"] else float("nan")
        lines += [
            "",
            "### Span length (predicted vs. gold, true positives only)",
            "",
            f"- predicted: mean {span_stats['pred_mean']:.2f}s, median {span_stats['pred_median']:.2f}s",
            f"- gold:      mean {span_stats['gold_mean']:.2f}s, median {span_stats['gold_median']:.2f}s "
            f"(n={span_stats['n']})",
            f"- ratio (predicted/gold): **{ratio:.2f}x**",
            "",
            "Ratio well above 1.0 means predicted spans are systematically wider than gold --",
            "tighten further (shorter quotes, a smaller fine-trim target duration). Ratio well",
            "below 1.0 means spans are being clipped too aggressively. A ratio near 1.0 with",
            "tIoU still low points at a placement problem instead -- right length, wrong position.",
        ]

    if statistics.latencies_ms:
        sorted_lat = sorted(statistics.latencies_ms)
        worst = sorted_lat[-1]
        lines += [
            "",
            "## Round trip",
            "",
            f"- worst case: {worst:.0f}ms ({worst / 60000:.0%} of the 60s per-request budget)",
            f"- mean: {sum(sorted_lat) / len(sorted_lat):.0f}ms",
        ]

    lines += ["", "## Worst offenders", "", "### Missed positives (wrong on both halves of the score)", ""]
    lines += _table(_worst_missed_positives(details, worst_n))

    lines += ["", "### hard_negative false positives (topical-similarity trap)", ""]
    lines += _table(_worst_hard_negative_false_positives(details, worst_n))

    lines += ["", "### Worst-localized true positives (answered right, span wrong)", ""]
    lines += _table(_worst_tiou_true_positives(details, worst_n))

    return "\n".join(lines)


def save_plots(statistics, details: List[DetailRow], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    tious = list(statistics.tious)   # includes the 0s for missed/no-span positives -- this IS the scored set
    axes[0].hist(tious, bins=20, range=(0, 1))
    axes[0].set_title("tIoU over annotated-yes questions\n(0 = missed or no span)")
    axes[0].set_xlabel("tIoU")

    types = ("positive", "hard_negative", "off_topic")
    accs = [
        statistics.by_type.get(t, [0, 1])[0] / max(1, statistics.by_type.get(t, [0, 1])[1])
        for t in types
    ]
    axes[1].bar(types, accs)
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, linestyle="--", linewidth=1, color="gray")
    axes[1].set_title("Accuracy by question type")

    if statistics.latencies_ms:
        axes[2].hist(statistics.latencies_ms, bins=20)
        axes[2].axvline(60000, linestyle="--", linewidth=1, color="red", label="60s budget")
        axes[2].set_title("Per-conversation latency (ms)")
        axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=official.DEFAULT_URL)
    parser.add_argument("--worst-n", type=int, default=10, help="examples per 'worst offenders' section")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not official.wait_for_endpoint(args.url):
        logger.error("Nothing answering at %s. Start it with 'python api.py' first.", args.url)
        return 1

    logger.info("Replaying the 39 training conversations with per-question detail ...")
    statistics, details = replay_with_detail(args.url, verbose=args.verbose)
    print(statistics.report())

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_short_sha()
    name = f"{timestamp}_{sha}"

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_md = build_report(statistics, details, args.worst_n, name)
    (reports_dir / f"analyze_{name}.md").write_text(report_md)

    report_json = {
        "name": name,
        "git_commit": sha,
        "score": statistics.final_score,
        "accuracy": statistics.accuracy,
        "mean_tiou": statistics.mean_tiou,
        "mean_tiou_answered_yes": statistics.mean_tiou_answered_yes,
        "missing_spans": statistics.missing_spans,
        "by_type": {k: {"correct": v[0], "total": v[1]} for k, v in statistics.by_type.items()},
        "latencies_ms": statistics.latencies_ms,
        "aborted": statistics.aborted,
        "timeouts": statistics.timeouts,
        "details": [
            {
                "audio_filename": d.audio_filename,
                "question_id": d.question_id,
                "question": d.question,
                "question_type": d.question_type,
                "label": d.label,
                "prediction": d.prediction,
                "gold_span": list(d.gold_span) if d.gold_span else None,
                "predicted_span": list(d.predicted_span) if d.predicted_span else None,
                "tiou": d.tiou,
            }
            for d in details
        ],
    }
    (reports_dir / f"analyze_{name}.json").write_text(json.dumps(report_json, indent=2))
    logger.info("Wrote reports/analyze_%s.md and .json", name)

    try:
        save_plots(statistics, details, reports_dir / f"analyze_{name}_plots.png")
        logger.info("Wrote reports/analyze_%s_plots.png", name)
    except ImportError:
        logger.warning("matplotlib not installed (pip install matplotlib) -- skipped plots, .md/.json are still complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
