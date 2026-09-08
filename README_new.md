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

## Next Steps

- Temporal smoothing (reduce frame-to-frame noise)
- Object tracking (maintain person identity)
- Zone persistence (require condition to persist N frames)
- Performance comparison vs baseline

## References

- Redmon et al., "You Only Look Once" (YOLO), CVPR 2016
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com/

## Author

Aditya Raj Jain  
Manipal University Jaipur  
Supervised by: Dr. Mohit Kushwaha
