from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
cam = cv2.VideoCapture(0)

frameCount = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frameCount += 1
    if frameCount % 2 != 0:
        continue

    camQuality = model(frame, imgsz=320, verbose=False)
    annotated = camQuality[0].plot()

    cv2.imshow("ASL Detection", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()