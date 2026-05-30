from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="BGpuuvjOigJfTT7mNdKY")
project = rf.workspace("sahiths-workspace").project("asl-dataset-p9yw8-zt56v")
dataset = project.version(1).download("yolov8")

model = YOLO("yolov8n.pt")
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=320,
    batch=8
)