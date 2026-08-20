import onnx
import onnxruntime as ort

MODEL_PATH = "models/yolov8n.onnx"


def main():
    print(f"Loading: {MODEL_PATH}")

    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model)

    print("ONNX model validation: OK")

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    print("\nInputs:")
    for inp in session.get_inputs():
        print(f"  name : {inp.name}")
        print(f"  shape: {inp.shape}")
        print(f"  type : {inp.type}")

    print("\nOutputs:")
    for output in session.get_outputs():
        print(f"  name : {output.name}")
        print(f"  shape: {output.shape}")
        print(f"  type : {output.type}")


if __name__ == "__main__":
    main()