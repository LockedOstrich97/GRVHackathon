from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")
cam = cv2.VideoCapture(0)


frameCount = 0


while True:
    ret, frame = cam.read()
    if not ret:
        break


    frameCount+=1
    if frameCount%2 !=0:
        continue


    camQuaility = model(frame, imgsz = 320, verbose = False)
    annotated = camQuaility[0].plot()


    cv2.imshow("Yolo Detection", annotated)
    if cv2.waitKey(1) == ord("q"):
        break




cam.release()
cv2.destroyAllWindows()

