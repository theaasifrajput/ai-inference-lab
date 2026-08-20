# Inference

The inference pipeline converts an input image into a tensor, executes the ONNX model, and converts the raw model output into application-level detections.

## Inference Pipeline

```text
Input Image
    ↓
Image Decode
    ↓
Resize / Letterbox
    ↓
RGB
    ↓
HWC → CHW
    ↓
Normalization
    ↓
NCHW FP32 Tensor
    ↓
ONNX Runtime
    ↓
Raw Model Output
    ↓
Post-processing
    ↓
Detections


Input Image

The pipeline accepts an image as input.

Supported formats:

JPEG
PNG
WebP

The image is decoded before preprocessing.

Preprocessing

The input image is converted into the tensor format expected by YOLOv8.

The preprocessing steps are:

Convert the image to RGB.
Resize while preserving the original aspect ratio.
Apply letterboxing to produce a 640 × 640 image.
Convert the image from HWC to CHW layout.
Convert pixel values to float32.
Normalize pixel values to the [0, 1] range.
Add the batch dimension.


The resulting tensor has:

Shape: [1, 3, 640, 640]
Type:  float32
Layout: NCHW

Letterboxing

The original image may not have a 1:1 aspect ratio.

Instead of stretching the image, we preserve its aspect ratio and add padding.

Original Image
      ↓
Resize while preserving aspect ratio
      ↓
Add padding
      ↓
640 × 640

The scale and padding values are retained because they are required later to map detection coordinates back to the original image.

Tensor Conversion

Images are normally represented as:

H × W × C

which is the HWC layout.

The model expects:

N × C × H × W

which is the NCHW layout.

Therefore:

HWC
 ↓
CHW
 ↓
Add batch dimension
 ↓
NCHW

For YOLOv8:

[H, W, 3]
      ↓
[3, H, W]
      ↓
[1, 3, 640, 640]

Normalization

Pixel values are normally in the range:

0 → 255

They are converted to float32 and normalized:

pixel = pixel / 255.0

The resulting values are approximately:

0.0 → 1.0

ONNX Runtime

The preprocessed tensor is passed to ONNX Runtime.

The initial execution provider is:

CPUExecutionProvider

The inference session should be created once and reused for multiple requests rather than creating a new session for every image.

Conceptually:
Input Tensor
     ↓
ONNX Runtime
     ↓
YOLOv8 ONNX Model
     ↓
Raw Output

Raw Model Output

The model returns:

[1, 84, 8400]

The raw output is not directly suitable for the API response.

It must first be transformed into individual predictions.

For processing, the output can be represented as:

[8400, 84]

where each row represents one candidate prediction.

Post-processing

Post-processing performs:

Confidence filtering.
Bounding-box coordinate conversion.
Coordinate mapping back to the original image.
Non-maximum suppression.
Detection object creation.

The final result contains information such as:

label
confidence
bounding box

Latency

Inference should eventually be measured as separate stages:

Preprocessing
     ↓
Model Execution
     ↓
Post-processing
     ↓
End-to-End

This allows us to identify where optimization provides the greatest benefit.

For example:

preprocess_ms
inference_ms
postprocess_ms
total_ms

These measurements will also be used as the baseline for the future C++ implementation.


## Running Inference

The inference pipeline can be executed directly from the command line.

From the `python` directory:

```bash
python -m scripts.infer tests/data/person.jpg