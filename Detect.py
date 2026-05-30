from collections import Counter
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
cam = cv2.VideoCapture(0)

frame_count = 0
detections = []

# FIX 1: Shrink sample size so the memory clears out faster
SAMPLE_SIZE = 10  
CONFIDENCE = 0.5
none_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    results = model(frame, imgsz=320, verbose=False, stream=True)

    best_box = None
    best_label = "None"
    best_conf = 0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf > best_conf and conf >= CONFIDENCE:
                best_conf = conf
                best_label = model.names[int(box.cls)]
                best_box = box.xyxy.tolist()

    # FIX 2: Clear memory if the hand drops or changes completely
    if best_label == "None":
        none_counter += 1
        if none_counter >= 3:  # Clear history after 3 consecutive empty frames
            detections.clear()
    else:
        none_counter = 0  # Reset counter if a valid sign is seen

    detections.append(best_label)
    if len(detections) > SAMPLE_SIZE:
        detections.pop(0)

    active_detections = [d for d in detections if d != "None"]
    if active_detections:
        most_common_label, count = Counter(active_detections).most_common(1)[0]
    else:
        most_common_label = "Waiting..."

    display = frame.copy()

    if best_box and most_common_label != "Waiting...":
        # Extract coordinates safely from the nested list
        x1, y1, x2, y2 = map(int, best_box[0])

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{most_common_label} ({best_conf:.2f})"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(display, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(display, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(display, f"Current Sign: {most_common_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("ASL Detection", display)

    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
