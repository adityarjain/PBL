from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# SETTINGS
PERSON_THRESHOLD = 6     # Global crowd alert
GRID_SIZE = 4            # 4x4 grid
ZONE_THRESHOLD = 3       # Per-cell overcrowding


def main(source=0):
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        h, w = frame.shape[:2]
        cell_w = w // GRID_SIZE
        cell_h = h // GRID_SIZE

        # Grid initialization
        grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]

        # YOLO Detection
        results = model(frame, conf=0.5, verbose=False)

        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person
                    person_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    # Center point
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Map to grid
                    gx = min(cx // cell_w, GRID_SIZE - 1)
                    gy = min(cy // cell_h, GRID_SIZE - 1)

                    grid[gy][gx] += 1

        # Draw grid lines
        for i in range(1, GRID_SIZE):
            cv2.line(frame, (i*cell_w, 0), (i*cell_w, h), (255,255,255), 1)
            cv2.line(frame, (0, i*cell_h), (w, i*cell_h), (255,255,255), 1)

        # 🔥 SMART ZONE COLORING
        overlay = frame.copy()
        zone_alert = False

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                count = grid[y][x]

                if count >= ZONE_THRESHOLD:
                    color = (0, 0, 255)  # RED (high density)
                    zone_alert = True
                elif count == 2:
                    color = (0, 255, 255)  # YELLOW (medium)
                elif count == 1:
                  continue   # ❌ skip green

                cv2.rectangle(
                    overlay,
                    (x*cell_w, y*cell_h),
                    ((x+1)*cell_w, (y+1)*cell_h),
                    color,
                    -1
                )

        # Blend overlay ONCE
        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

        # 🔥 ALERTS
        if person_count >= PERSON_THRESHOLD:
            cv2.putText(frame, "ALERT: HIGH CROWD", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if zone_alert:
            cv2.putText(frame, "ZONE OVERCROWD", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # FPS
        fps = 1 / (time.time() - start)

        # Display info
        cv2.putText(frame, f"Count: {person_count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (10,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("YOLO Crowd Monitoring - Stage 2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0)