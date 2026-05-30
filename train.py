from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("API_KEY")


rf = Roboflow(api_key=apiKey)
project = rf.workspace("sahiths-workspace").project("asl-dataset-p9yw8-zt56v")
dataset = project.version(1).download("yolov8")

model = YOLO("yolov8n.pt")
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=320,
    batch=8
)