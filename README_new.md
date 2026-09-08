# AI-Based Crowd Monitoring and Safety Alert System

A real-time crowd detection and density analysis system using YOLOv8.

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python yolo_demo.py
```

Press `q` to quit.

## How It Works

1. **Detection** - YOLOv8 detects people in real-time
2. **Counting** - Counts total people in frame
3. **Spatial Analysis** - Divides frame into 4×4 grid
4. **Density Classification** - Marks zones as safe/moderate/crowded
5. **Alerts** - Triggers when thresholds exceeded

## Thresholds

- `PERSON_THRESHOLD = 6` - Global alert if total >= 6 people
- `ZONE_THRESHOLD = 3` - Zone alert if any cell >= 3 people
- Confidence threshold = 0.5

## Current Status

✓ YOLO detection working  
✓ Grid-based density analysis working  
✓ Real-time alerts working  
⚠️ Performance metrics being measured  

## Evaluation

### Baseline Performance Measurement

Run baseline evaluation to measure FPS, inference time, and detection counts on your hardware:

```bash
python evaluate_baseline.py
```

**Arguments:**
- `source` (default: 0 = webcam, or path to video file)
- `duration_seconds` (default: 60)
- `confidence` (default: 0.5)

**Example with video file:**
```bash
python evaluate_baseline.py "path/to/video.mp4" 60 0.5
```

**Output:**
- Console summary with metrics
- Results saved to `baseline_results.json`

Measures:
- Average FPS on your hardware
- Inference time per frame (ms)
- Total people detected
- Memory usage
- Test conditions (resolution, date, source)

### Confidence Threshold Optimization

Test different confidence thresholds to find optimal balance between detection accuracy and false positives:

```bash
python experiment_confidence_thresholds.py
```

**Tests thresholds:** 0.3, 0.5, 0.7

**Output:**
- Console table with results
- Results saved to `conf_threshold_results.csv`

**Tradeoffs:**
- **0.3 (low):** More detections, more false positives
- **0.5 (balanced):** Default, good balance
- **0.7 (high):** Fewer false positives, might miss people

## Phase 2: Temporal Smoothing Experiment

**Research question:** Does moving-average smoothing of detection counts reduce
alert flicker/noise without a significant FPS or latency cost?

This is run as a controlled experiment, not assumed to work — `yolo_demo.py`
is kept unmodified as the reference baseline; `yolo_demo_v2_smoothed.py` is an
isolated copy with only the smoothing logic added (5-frame moving average on
both the global person count and each grid zone's count).

### Quick sanity check (no video/model needed)

```bash
python test_smoothing.py
```

### Run the experiment

```bash
# 1. Measure baseline (raw, unsmoothed) alert behavior
python measure_system_metrics.py --mode baseline --source videos/crowd.mp4 --out baseline_measurements.csv

# 2. Measure smoothed alert behavior on the SAME video
python measure_system_metrics.py --mode smoothed --source videos/crowd.mp4 --out v2_measurements.csv

# 3. Compare
python compare_baseline_vs_v2.py --baseline baseline_measurements.csv --v2 v2_measurements.csv
```

Produces `comparison_report.md` with measured (not assumed) results:
alert event counts, transient/flickering alert rate, alert duration,
detection variability, and latency — for both the global crowd alert and
the per-zone overcrowd alert.

**Note:** `videos/crowd.mp4` is real footage without per-frame ground-truth
annotations, so we report alert frequency/duration/stability rather than a
"false positive rate" we can't actually substantiate.

### Visual check

```bash
python yolo_demo_v2_smoothed.py   # webcam, side-by-side comparable to yolo_demo.py
```

## Next Steps

- Evidence-based decision on frame persistence/hysteresis (Tier 2), based on
  the Phase 2 measurement results above
- Object tracking (maintain person identity) — deferred until justified by data

## References

- Redmon et al., "You Only Look Once" (YOLO), CVPR 2016
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com/

## Author

Aditya Raj Jain  
Manipal University Jaipur  
Supervised by: Dr. Mohit Kushwaha
