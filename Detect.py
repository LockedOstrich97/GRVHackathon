from ultralytics import YOLO
import cv2
from collections import Counter
import time


model_local = YOLO("runs/detect/letters-3/weights/best.pt")
cam = cv2.VideoCapture(0)

# Video recording setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = time.strftime("%Y%m%d-%H%M%S")
out = cv2.VideoWriter(f"recording_{timestamp}.mp4", fourcc, 20.0, (640, 480))
recording = False

frameCount = 0
detections = []
SAMPLE_SIZE = 30
CONFIDENCE = 0.5
last_printed = None

print("Press R to start/stop recording, Q to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frameCount += 1
    if frameCount % 30 != 0:
        continue

    results = model_local(frame, imgsz=320, verbose=False, max_det=1)

    # Only take the single highest confidence detection per frame
    best_label = None
    best_conf = 0

    for box in results[0].boxes:
        conf = float(box.conf)
        label = model_local.names[int(box.cls)]
        if conf > best_conf and conf >= CONFIDENCE:
            best_conf = conf
            best_label = label

    if best_label:
        detections.append(best_label)

    if len(detections) > SAMPLE_SIZE:
        detections.pop(0)

    # Pick the single most common detection
    if detections:
        most_common_label, count = Counter(detections).most_common(1)[0]
        consistency = count / len(detections)

        # Only print when the sign changes
        if most_common_label != last_printed:
            print(f"Sign: {most_common_label} ({consistency:.0%} consistent)")
            last_printed = most_common_label
    else:
        most_common_label = "Waiting..."
        consistency = 0

    # Draw frame with one box
    display = results[0].plot()

    # Show averaged result at top
    cv2.putText(display, f"Sign: {most_common_label}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(display, f"Confidence: {consistency:.0%}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show recording indicator
    if recording:
        cv2.circle(display, (620, 30), 15, (0, 0, 255), -1)  # red dot
        cv2.putText(display, "REC", (580, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        out.write(display)  # save frame to video

    cv2.imshow("ASL Detection", display)

    key = cv2.waitKey(33) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        recording = not recording
        if recording:
            print(f"Recording started -> recording_{timestamp}.mp4")
        else:
            print("Recording stopped")

out.release()
cam.release()
cv2.destroyAllWindows()