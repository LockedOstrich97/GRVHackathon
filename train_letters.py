from roboflow import Roboflow
from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()  

apiKey = os.getenv("API_KEY")

rf = Roboflow(api_key=apiKey)
project = rf.workspace("sahiths-workspace").project("asl-79h3m-6enj5")
version = project.version(1)
dataset = version.download("yolov8")

model = YOLO("yolov8n.pt")
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=320,
    batch=8,
    name="letters"
)