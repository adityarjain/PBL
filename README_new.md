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

## Next Steps

- Baseline FPS/accuracy measurement
- Confidence threshold tuning
- Temporal smoothing
- Object tracking

## References

- Redmon et al., "You Only Look Once" (YOLO), CVPR 2016
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com/

## Author

Aditya Raj Jain  
Manipal University Jaipur  
Supervised by: Dr. Mohit Kushwaha
