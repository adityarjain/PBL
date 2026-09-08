from ultralytics import YOLO
import cv2
import time
import csv
from pathlib import Path

def test_confidence_threshold(source=0, confidence=0.5, duration_seconds=30):
    """
    Test a specific confidence threshold and return metrics

    Args:
        source: 0 for webcam, or path to video file
        confidence: Confidence threshold to test
        duration_seconds: How long to run test
    """

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError("Could not open video source")

    total_frames = 0
    total_people = 0
    total_inference_time = 0.0
    start_time = time.time()
    zone_alerts = 0

    GRID_SIZE = 4
    ZONE_THRESHOLD = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            cell_w = w // GRID_SIZE
            cell_h = h // GRID_SIZE

            grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]

            # Inference
            inf_start = time.time()
            results = model(frame, conf=confidence, verbose=False)
            inf_time = (time.time() - inf_start) * 1000

            # Count people and assign to grid
            person_count = 0
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        gx = min(cx // cell_w, GRID_SIZE - 1)
                        gy = min(cy // cell_h, GRID_SIZE - 1)
                        grid[gy][gx] += 1

            # Count zone alerts
            frame_zone_alerts = 0
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    if grid[y][x] >= ZONE_THRESHOLD:
                        frame_zone_alerts += 1

            total_frames += 1
            total_people += person_count
            total_inference_time += inf_time
            zone_alerts += frame_zone_alerts

            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = total_frames / total_time if total_time > 0 else 0
    avg_inference_ms = total_inference_time / total_frames if total_frames > 0 else 0

    return {
        "confidence_threshold": confidence,
        "total_detections": total_people,
        "avg_fps": round(avg_fps, 2),
        "avg_inference_ms": round(avg_inference_ms, 2),
        "zones_alert_count": zone_alerts,
    }

def run_experiment(source=0, duration_per_threshold=30):
    """
    Test multiple confidence thresholds and save results to CSV

    Args:
        source: 0 for webcam, or path to video file
        duration_per_threshold: Seconds to test each threshold
    """

    thresholds = [0.3, 0.5, 0.7]

    print("Loading YOLOv8 model...")
    YOLO("yolov8n.pt")  # Preload model

    print(f"Running confidence threshold experiment...")
    print(f"Testing thresholds: {thresholds}")
    print(f"Duration per threshold: {duration_per_threshold}s\n")

    results = []

    for threshold in thresholds:
        print(f"Testing confidence={threshold}...")
        result = test_confidence_threshold(
            source=source,
            confidence=threshold,
            duration_seconds=duration_per_threshold
        )
        results.append(result)
        print(f"  ✓ Detections: {result['total_detections']}, FPS: {result['avg_fps']}, Zone Alerts: {result['zones_alert_count']}\n")

    # Save to CSV
    output_file = "conf_threshold_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["confidence_threshold", "total_detections", "avg_fps", "avg_inference_ms", "zones_alert_count"]
        )
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print("="*70)
    print("=== CONFIDENCE THRESHOLD EXPERIMENT RESULTS ===")
    print("="*70)
    print(f"{'Threshold':<12} {'Detections':<15} {'FPS':<12} {'Inference (ms)':<18} {'Zone Alerts':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['confidence_threshold']:<12} {r['total_detections']:<15} {r['avg_fps']:<12} {r['avg_inference_ms']:<18} {r['zones_alert_count']:<12}")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    print("\nAnalysis:")
    print("- Lower threshold (0.3) detects more people but may have false positives")
    print("- Higher threshold (0.7) is more selective, better precision but may miss people")
    print("- 0.5 is balanced default")

if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else 0
    source = int(source) if source.isdigit() else source

    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30

    run_experiment(source=source, duration_per_threshold=duration)
