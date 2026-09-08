"""
Headless measurement harness for crowd-monitoring alert behavior.

Replicates the exact detection/grid/alert logic from yolo_demo.py (baseline mode)
or with moving-average smoothing added (smoothed mode) -- WITHOUT any display,
so it can run unattended on a fixed video file for repeatable measurement.

yolo_demo.py itself is never imported or modified; this script re-implements
the same algorithm so both modes can be measured under identical conditions
with per-frame CSV logging.

Usage:
    python measure_system_metrics.py --mode baseline --source videos/crowd.mp4 --out baseline_measurements.csv
    python measure_system_metrics.py --mode smoothed --source videos/crowd.mp4 --out v2_measurements.csv
"""

import argparse
import csv
import time
from collections import deque

from ultralytics import YOLO
import cv2

# Must match yolo_demo.py exactly (baseline reference values)
PERSON_THRESHOLD = 6
GRID_SIZE = 4
ZONE_THRESHOLD = 3
CONFIDENCE = 0.5
SMOOTHING_WINDOW = 5  # only used in "smoothed" mode


def moving_average(buffer):
    return sum(buffer) / len(buffer) if buffer else 0.0


def run_measurement(source, mode, output_csv, max_frames=None):
    print(f"Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {source}")
    print(f"Mode: {mode}")
    print(f"Video FPS: {video_fps:.2f}, Total frames: {total_frames_in_video}")

    # Smoothing state (only used in smoothed mode)
    person_history = deque(maxlen=SMOOTHING_WINDOW)
    zone_history = [[deque(maxlen=SMOOTHING_WINDOW) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    rows = []
    frame_idx = 0
    run_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cell_w = w // GRID_SIZE
        cell_h = h // GRID_SIZE
        grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

        t0 = time.time()
        results = model(frame, conf=CONFIDENCE, verbose=False)
        inference_ms = (time.time() - t0) * 1000

        raw_person_count = 0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person class
                    raw_person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    gx = min(cx // cell_w, GRID_SIZE - 1)
                    gy = min(cy // cell_h, GRID_SIZE - 1)
                    grid[gy][gx] += 1

        if mode == "smoothed":
            person_history.append(raw_person_count)
            eval_count = moving_average(person_history)
            eval_grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    zone_history[gy][gx].append(grid[gy][gx])
                    eval_grid[gy][gx] = moving_average(zone_history[gy][gx])
        elif mode == "baseline":
            eval_count = raw_person_count
            eval_grid = grid
        else:
            raise ValueError(f"Unknown mode: {mode}")

        global_alert = eval_count >= PERSON_THRESHOLD
        zone_alert = any(
            eval_grid[gy][gx] >= ZONE_THRESHOLD
            for gy in range(GRID_SIZE) for gx in range(GRID_SIZE)
        )
        num_zones_alerting = sum(
            1 for gy in range(GRID_SIZE) for gx in range(GRID_SIZE)
            if eval_grid[gy][gx] >= ZONE_THRESHOLD
        )

        rows.append({
            "frame": frame_idx,
            "time_sec": round(frame_idx / video_fps, 3) if video_fps > 0 else frame_idx,
            "raw_person_count": raw_person_count,
            "eval_person_count": round(eval_count, 3),
            "global_alert": int(global_alert),
            "zone_alert": int(zone_alert),
            "num_zones_alerting": num_zones_alerting,
            "inference_ms": round(inference_ms, 3),
        })

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx} frames...")
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    total_time = time.time() - run_start

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print(f"\nDone. {frame_idx} frames processed in {total_time:.1f}s (avg {avg_fps:.1f} FPS incl. overhead)")
    print(f"Results written to: {output_csv}")

    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure crowd-monitoring alert behavior (headless)")
    parser.add_argument("--mode", choices=["baseline", "smoothed"], required=True,
                         help="baseline = raw yolo_demo.py logic; smoothed = with moving-average smoothing")
    parser.add_argument("--source", default="videos/crowd.mp4", help="Video file path")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed (for quick tests)")
    args = parser.parse_args()

    out_path = args.out or f"{args.mode}_measurements.csv"
    run_measurement(args.source, args.mode, out_path, args.max_frames)
