from __future__ import annotations

import numpy as np

from services.inference.models import Detection


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from:

        [center_x, center_y, width, height]

    to:

        [x1, y1, x2, y2]
    """

    result = boxes.copy()

    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return result


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between one box and multiple boxes.

    Boxes use [x1, y1, x2, y2] format.
    """

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0.0, x2 - x1)
        * np.maximum(0.0, y2 - y1)
    )

    box_area = (
        (box[2] - box[0])
        * (box[3] - box[1])
    )

    boxes_area = (
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1])
    )

    union = box_area + boxes_area - intersection

    return intersection / np.maximum(union, 1e-6)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[int]:
    """
    Perform non-maximum suppression.

    Returns indices of boxes that should be kept.
    """

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while len(order) > 0:
        current = order[0]
        keep.append(int(current))

        if len(order) == 1:
            break

        remaining = order[1:]

        overlaps = iou(
            boxes[current],
            boxes[remaining],
        )

        order = remaining[overlaps <= threshold]

    return keep


def postprocess(
    output: np.ndarray,
    confidence_threshold: float,
    nms_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_width: int,
    original_height: int,
) -> list[Detection]:
    """
    Convert raw YOLOv8 output into person detections.

    Args:
        output:
            Raw model output with shape [1, 84, 8400].

        confidence_threshold:
            Minimum person confidence.

        nms_threshold:
            IoU threshold for NMS.

        scale:
            Letterbox resize scale.

        pad_x:
            Horizontal letterbox padding.

        pad_y:
            Vertical letterbox padding.

        original_width:
            Original image width.

        original_height:
            Original image height.
    """

    # [1, 84, 8400] -> [8400, 84]
    predictions = np.squeeze(output, axis=0).T

    # YOLOv8:
    #
    # 0:4  -> x, y, w, h
    # 4    -> class 0 (person)
    #
    boxes = predictions[:, :4]
    scores = predictions[:, 4]

    # Confidence filtering
    mask = scores >= confidence_threshold

    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return []

    # xywh -> xyxy
    boxes = xywh_to_xyxy(boxes)

    # Convert from 640x640 letterboxed coordinates
    # back to original image coordinates.
    boxes[:, [0, 2]] = (
        (boxes[:, [0, 2]] - pad_x) / scale
    )

    boxes[:, [1, 3]] = (
        (boxes[:, [1, 3]] - pad_y) / scale
    )

    # Clip to original image boundaries.
    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0,
        original_width,
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0,
        original_height,
    )

    # Non-maximum suppression.
    keep = nms(
        boxes,
        scores,
        nms_threshold,
    )

    return [
        Detection(
            label="person",
            confidence=float(scores[index]),
            box=[
                round(float(value), 1)
                for value in boxes[index]
            ],
        )
        for index in keep
    ]
