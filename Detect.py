from ultralytics import YOLO
import cv2
from collections import Counter

model = YOLO("runs/detect/train/weights/best.pt")
cam = cv2.VideoCapture(0)

frameCount = 0
detections = []
SAMPLE_SIZE = 30
CONFIDENCE = 0.5
last_printed = None

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frameCount += 1
    if frameCount % 5 != 0:
        continue

    results = model(frame, imgsz=320, verbose=False, max_det=1)

    # Only take the single highest confidence detection per frame
    best_label = None
    best_conf = 0

    for box in results[0].boxes:
        conf = float(box.conf)
        label = model.names[int(box.cls)]
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

    cv2.imshow("ASL Detection", display)
    if cv2.waitKey(33) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()