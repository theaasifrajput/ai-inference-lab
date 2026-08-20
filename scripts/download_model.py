from pathlib import Path
import shutil

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolov8n.pt"
TARGET = MODEL_DIR / MODEL_NAME


def main():
    if TARGET.exists():
        print(f"Already exists: {TARGET}")
        return

    print(f"Downloading {MODEL_NAME}...")

    model = YOLO(MODEL_NAME)

    source = Path(model.ckpt_path)

    if source.resolve() != TARGET.resolve():
        shutil.copy2(source, TARGET)

    print(f"Saved model to: {TARGET}")


if __name__ == "__main__":
    main()