from ultralytics import YOLO
import cv2
import time
import json
from pathlib import Path
import psutil
import os

def get_memory_usage():
    """Get memory usage of current process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def evaluate_baseline(source=0, duration_seconds=60, confidence=0.5):
    """
    Evaluate baseline performance of YOLOv8 crowd detection

    Args:
        source: 0 for webcam, or path to video file
        duration_seconds: How long to run evaluation
        confidence: Detection confidence threshold
    """

    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("ERROR: Could not open video source")
        return None

    # Get video properties
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: {width}x{height} @ {fps_input} FPS (input)")
    print(f"Running evaluation for ~{duration_seconds} seconds...\n")

    # Metrics
    total_frames = 0
    total_people = 0
    total_inference_time = 0.0
    start_time = time.time()
    inference_times = []

    initial_memory = get_memory_usage()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            inf_start = time.time()
            results = model(frame, conf=confidence, verbose=False)
            inf_time = (time.time() - inf_start) * 1000  # Convert to ms

            # Count people
            person_count = 0
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        person_count += 1

            total_frames += 1
            total_people += person_count
            total_inference_time += inf_time
            inference_times.append(inf_time)

            # Check if duration exceeded
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                break

            # Progress indicator
            if total_frames % 30 == 0:
                current_fps = total_frames / elapsed
                print(f"Frame {total_frames} | Elapsed: {elapsed:.1f}s | FPS: {current_fps:.1f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Calculate metrics
    total_time = time.time() - start_time
    avg_fps = total_frames / total_time if total_time > 0 else 0
    avg_inference_ms = total_inference_time / total_frames if total_frames > 0 else 0
    avg_people_per_frame = total_people / total_frames if total_frames > 0 else 0
    final_memory = get_memory_usage()

    # Results dictionary
    results_dict = {
        "model": "yolov8n",
        "confidence_threshold": confidence,
        "input_resolution": f"{width}x{height}",
        "total_frames": total_frames,
        "total_time_seconds": round(total_time, 2),
        "average_fps": round(avg_fps, 2),
        "average_inference_ms": round(avg_inference_ms, 2),
        "total_people_detected": total_people,
        "average_people_per_frame": round(avg_people_per_frame, 2),
        "memory_usage_mb_start": round(initial_memory, 2),
        "memory_usage_mb_end": round(final_memory, 2),
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_source": "webcam" if source == 0 else str(source),
    }

    # Save to JSON
    output_file = "baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("=== YOLO BASELINE EVALUATION ===")
    print("="*50)
    print(f"Model:                  {results_dict['model']}")
    print(f"Confidence Threshold:   {results_dict['confidence_threshold']}")
    print(f"Resolution:             {results_dict['input_resolution']}")
    print(f"Duration:               {results_dict['total_time_seconds']}s")
    print(f"Total Frames:           {results_dict['total_frames']}")
    print(f"Average FPS:            {results_dict['average_fps']}")
    print(f"Average Inference:      {results_dict['average_inference_ms']}ms")
    print(f"Total People Detected:  {results_dict['total_people_detected']}")
    print(f"Average People/Frame:   {results_dict['average_people_per_frame']}")
    print(f"Memory (start/end):     {results_dict['memory_usage_mb_start']}/{results_dict['memory_usage_mb_end']} MB")
    print(f"Test Date:              {results_dict['test_date']}")
    print("="*50)
    print(f"\nResults saved to: {output_file}")

    return results_dict

if __name__ == "__main__":
    import sys

    # Arguments: [source] [duration] [confidence]
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    source = int(source) if source.isdigit() else source

    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    evaluate_baseline(source=source, duration_seconds=duration, confidence=confidence)
