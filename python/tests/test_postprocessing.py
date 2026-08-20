import numpy as np
import pytest

from services.inference.postprocessing import (
    iou,
    nms,
    postprocess,
    xywh_to_xyxy,
)


def test_xywh_to_xyxy():
    boxes = np.array(
        [
            [100, 100, 40, 20],
        ],
        dtype=np.float32,
    )

    result = xywh_to_xyxy(boxes)

    expected = np.array(
        [
            [80, 90, 120, 110],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected)


def test_iou_identical_boxes():
    box = np.array(
        [0, 0, 100, 100],
        dtype=np.float32,
    )

    boxes = np.array(
        [
            [0, 0, 100, 100],
        ],
        dtype=np.float32,
    )

    result = iou(box, boxes)

    np.testing.assert_allclose(result, [1.0])


def test_nms_removes_overlapping_box():
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [10, 10, 90, 90],
        ],
        dtype=np.float32,
    )

    scores = np.array(
        [0.9, 0.8],
        dtype=np.float32,
    )

    keep = nms(
        boxes,
        scores,
        threshold=0.5,
    )

    assert keep == [0]


def test_postprocess_person_detection():
    output = np.zeros(
        (1, 84, 8400),
        dtype=np.float32,
    )

    # Candidate prediction 0
    #
    # YOLOv8 layout:
    # 0 -> x
    # 1 -> y
    # 2 -> width
    # 3 -> height
    # 4 -> person score (COCO class 0)

    output[0, 0, 0] = 320.0
    output[0, 1, 0] = 320.0
    output[0, 2, 0] = 100.0
    output[0, 3, 0] = 200.0
    output[0, 4, 0] = 0.90

    detections = postprocess(
        output=output,
        confidence_threshold=0.5,
        nms_threshold=0.5,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        original_width=640,
        original_height=640,
    )

    assert len(detections) == 1

    detection = detections[0]

    assert detection.label == "person"
    assert detection.confidence == pytest.approx(0.90)
    assert detection.box == [270.0, 220.0, 370.0, 420.0]