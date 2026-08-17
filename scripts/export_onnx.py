"""Export YOLOv8 nano to the ONNX artifact consumed by the Phase-1 service."""
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
artifact = Path(model.export(format="onnx", imgsz=640, dynamic=False, simplify=True))
Path("models").mkdir(exist_ok=True)
artifact.replace("models/yolov8n.onnx")
print("Exported models/yolov8n.onnx")
