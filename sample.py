import cv2
import time
import os

PERSON_COUNT_THRESHOLD = 6
OCCUPIED_RATIO_THRESHOLD = 0.20

ALERT_SAVE_PATH = "alerts"
os.makedirs(ALERT_SAVE_PATH, exist_ok=True)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def compute_occupied_area_ratio(boxes, fw, fh):
    if not boxes:
        return 0.0
    total = 0
    for (x, y, w, h) in boxes:
        total += w * h
    return total / (fw * fh)

def detect_people(frame):
    rects, weights = hog.detectMultiScale(frame,
                                          winStride=(8,8),
                                          padding=(8,8),
                                          scale=1.05)
    filtered = [(x,y,w,h) for (x,y,w,h),wgt in zip(rects,weights) if wgt > 0.5]
    return filtered

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: cannot open camera or video")
        return

    fps_time = time.time()
    fps_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        frame_small = cv2.resize(frame, (640, 480))
        fh, fw = frame_small.shape[:2]

        t0 = time.time()
        boxes = detect_people(frame_small)
        infer_ms = (time.time() - t0) * 1000

        person_count = len(boxes)
        occ_ratio = compute_occupied_area_ratio(boxes, fw, fh)

        for (x,y,w,h) in boxes:
            cv2.rectangle(frame_small, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.putText(frame_small, f"Count: {person_count}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame_small, f"OccRatio: {occ_ratio:.2f}", (10,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.putText(frame_small, f"Infer: {infer_ms:.0f}ms", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        alerts = []
        if person_count >= PERSON_COUNT_THRESHOLD:
            alerts.append("HIGH_PERSON_COUNT")
        if occ_ratio >= OCCUPIED_RATIO_THRESHOLD:
            alerts.append("HIGH_DENSITY")

        if alerts:
            alert_text = "ALERT: " + ",".join(alerts)
            cv2.putText(frame_small, alert_text, (10,110),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 2)
            ts = int(time.time())
            fname = os.path.join(ALERT_SAVE_PATH, f"alert_{ts}.jpg")
            cv2.imwrite(fname, frame_small)
            print("ALERT saved:", fname, alerts)

        fps_counter += 1
        if time.time() - fps_time >= 1:
            cv2.putText(frame_small, f"FPS: {fps_counter}", (10,95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            fps_counter = 0
            fps_time = time.time()

        cv2.imshow("Simple Crowd Demo", frame_small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n=== Crowd Monitor Demo ===")
    print("[W] Use Webcam")
    print("[V] Use Video File")
    mode = input("Select mode (W/V): ").strip().lower()

    if mode == "w":
        main(0)
    elif mode == "v":
        video_path = input("Enter video path (e.g., videos/crowd.mp4): ").strip()
        main(video_path)
    else:
        print("Invalid option. Exiting.")
