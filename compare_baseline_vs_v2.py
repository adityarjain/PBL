"""
Compare baseline (raw, no smoothing) vs v2 (moving-average smoothed) alert behavior,
using the CSVs produced by measure_system_metrics.py on the SAME video.

Reports honestly measured metrics -- no invented numbers, no ground-truth false-positive
claims for real (non-synthetic) video since per-frame ground truth isn't available.

Metrics computed (per user-specified methodology):
- Total alert events (rising edges: alert goes False -> True)
- Transient/flickering alert events (an alert run lasting <= 2 frames)
- Alert duration statistics (mean / median / max run length, in frames and seconds)
- Detection variability (frame-to-frame change in raw person count -- identical
  between baseline and v2 since same model/video, reported for context)
- FPS / inference latency (should be ~identical -- smoothing is post-detection)

Usage:
    python compare_baseline_vs_v2.py --baseline baseline_measurements.csv --v2 v2_measurements.csv
"""

import argparse
import csv
import statistics


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def alert_runs(rows, field):
    """Return list of run-lengths (in frames) for contiguous alert==1 stretches."""
    runs = []
    current = 0
    for row in rows:
        active = int(row[field]) == 1
        if active:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


def summarize_alerts(rows, field, video_fps):
    runs = alert_runs(rows, field)
    total_events = len(runs)
    transient_events = sum(1 for r in runs if r <= 2)
    frames_in_alert = sum(runs)
    total_frames = len(rows)

    return {
        "total_alert_events": total_events,
        "transient_events_le2frames": transient_events,
        "transient_fraction_of_events": round(transient_events / total_events, 3) if total_events else 0.0,
        "mean_alert_duration_frames": round(statistics.mean(runs), 2) if runs else 0.0,
        "median_alert_duration_frames": round(statistics.median(runs), 2) if runs else 0.0,
        "max_alert_duration_frames": max(runs) if runs else 0,
        "mean_alert_duration_sec": round(statistics.mean(runs) / video_fps, 3) if runs and video_fps else 0.0,
        "pct_frames_in_alert": round(100 * frames_in_alert / total_frames, 2) if total_frames else 0.0,
    }


def detection_variability(rows):
    counts = [int(r["raw_person_count"]) for r in rows]
    deltas = [abs(counts[i] - counts[i - 1]) for i in range(1, len(counts))]
    return {
        "mean_abs_frame_to_frame_delta": round(statistics.mean(deltas), 3) if deltas else 0.0,
        "stdev_person_count": round(statistics.stdev(counts), 3) if len(counts) > 1 else 0.0,
        "max_person_count": max(counts) if counts else 0,
        "min_person_count": min(counts) if counts else 0,
    }


def latency_stats(rows):
    times = [float(r["inference_ms"]) for r in rows]
    return {
        "mean_inference_ms": round(statistics.mean(times), 3) if times else 0.0,
        "median_inference_ms": round(statistics.median(times), 3) if times else 0.0,
        "max_inference_ms": round(max(times), 3) if times else 0.0,
    }


def compare(baseline_path, v2_path, video_fps=29.97, report_path="comparison_report.md"):
    baseline_rows = load_csv(baseline_path)
    v2_rows = load_csv(v2_path)

    if len(baseline_rows) != len(v2_rows):
        print(f"WARNING: frame count mismatch — baseline={len(baseline_rows)}, v2={len(v2_rows)}. "
              f"Comparing only the overlapping range.")
        n = min(len(baseline_rows), len(v2_rows))
        baseline_rows, v2_rows = baseline_rows[:n], v2_rows[:n]

    results = {
        "baseline": {
            "global_alert": summarize_alerts(baseline_rows, "global_alert", video_fps),
            "zone_alert": summarize_alerts(baseline_rows, "zone_alert", video_fps),
            "detection_variability": detection_variability(baseline_rows),
            "latency": latency_stats(baseline_rows),
        },
        "v2_smoothed": {
            "global_alert": summarize_alerts(v2_rows, "global_alert", video_fps),
            "zone_alert": summarize_alerts(v2_rows, "zone_alert", video_fps),
            "detection_variability": detection_variability(v2_rows),
            "latency": latency_stats(v2_rows),
        },
        "frames_compared": len(baseline_rows),
    }

    # Print to console
    print("=" * 70)
    print("BASELINE vs SMOOTHED (V2) — MEASURED COMPARISON")
    print("=" * 70)
    print(f"Frames compared: {results['frames_compared']}\n")

    for metric_group in ["global_alert", "zone_alert"]:
        print(f"--- {metric_group} ---")
        b = results["baseline"][metric_group]
        v = results["v2_smoothed"][metric_group]
        for key in b:
            print(f"  {key:38s} baseline={b[key]!s:>10}  v2={v[key]!s:>10}")
        print()

    print("--- detection variability (should be identical: same model/video) ---")
    for key in results["baseline"]["detection_variability"]:
        b = results["baseline"]["detection_variability"][key]
        v = results["v2_smoothed"]["detection_variability"][key]
        print(f"  {key:38s} baseline={b!s:>10}  v2={v!s:>10}")
    print()

    print("--- latency / FPS ---")
    for key in results["baseline"]["latency"]:
        b = results["baseline"]["latency"][key]
        v = results["v2_smoothed"]["latency"][key]
        print(f"  {key:38s} baseline={b!s:>10}  v2={v!s:>10}")
    print("=" * 70)

    # Write markdown report
    with open(report_path, "w") as f:
        f.write("# Phase 2 Comparison Report: Baseline vs Temporal Smoothing\n\n")
        f.write("## Methodology\n\n")
        f.write(f"- Same video processed with identical YOLO model, confidence (0.5), grid (4x4), thresholds.\n")
        f.write(f"- Baseline: raw per-frame detection count drives alert decisions (yolo_demo.py logic).\n")
        f.write(f"- V2: 5-frame moving average of detection count drives alert decisions "
                f"(yolo_demo_v2_smoothed.py logic).\n")
        f.write(f"- Frames compared: {results['frames_compared']}\n\n")
        f.write("**Note on false positives:** this is a real (non-synthetic) crowd video without "
                "per-frame ground truth annotations, so we do NOT claim a false-positive rate. "
                "Instead we report transient alert events (runs of <=2 frames), alert event counts, "
                "alert duration, and detection stability, per the measurement methodology.\n\n")

        for metric_group, label in [("global_alert", "Global Crowd Alert"), ("zone_alert", "Zone Overcrowd Alert")]:
            f.write(f"## {label}\n\n")
            f.write("| Metric | Baseline | V2 (Smoothed) |\n|---|---|---|\n")
            b = results["baseline"][metric_group]
            v = results["v2_smoothed"][metric_group]
            for key in b:
                f.write(f"| {key} | {b[key]} | {v[key]} |\n")
            f.write("\n")

        f.write("## Detection Variability (context — identical detections feed both systems)\n\n")
        f.write("| Metric | Baseline | V2 (Smoothed) |\n|---|---|---|\n")
        for key in results["baseline"]["detection_variability"]:
            b = results["baseline"]["detection_variability"][key]
            v = results["v2_smoothed"]["detection_variability"][key]
            f.write(f"| {key} | {b} | {v} |\n")
        f.write("\n")

        f.write("## Latency / Throughput\n\n")
        f.write("| Metric | Baseline | V2 (Smoothed) |\n|---|---|---|\n")
        for key in results["baseline"]["latency"]:
            b = results["baseline"]["latency"][key]
            v = results["v2_smoothed"]["latency"][key]
            f.write(f"| {key} | {b} | {v} |\n")
        f.write("\n")

        f.write("## Interpretation\n\n")
        f.write("_Fill in after reviewing the numbers above — report what the data actually shows, "
                "not the hypothesis. See CLAUDE_CODE_TASK.md guidance: report the real percentage, "
                "even if it's small, zero, or negative._\n")

    print(f"\nReport written to: {report_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs smoothed alert measurements")
    parser.add_argument("--baseline", default="baseline_measurements.csv")
    parser.add_argument("--v2", default="v2_measurements.csv")
    parser.add_argument("--video-fps", type=float, default=29.97)
    parser.add_argument("--report", default="comparison_report.md")
    args = parser.parse_args()

    compare(args.baseline, args.v2, args.video_fps, args.report)
