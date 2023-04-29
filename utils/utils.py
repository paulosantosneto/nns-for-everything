import numpy as np
from typing import Optional
import torch

def assert_input_array(bbox):
    
    try:
        assert len(bbox) == 4, bbox.dtype == np.float32
    except AssertionError as e:
        e.args += ('length isn ot compatible', 'type of data is not compatible')

def IOU(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate intersection over union between two bounding boxes.

    @param bbox1: bounding boxes 1 [x1, y1, x2, y2].
    @param bbox2: bounding boxes 2 [x1, y1, x2, y2].
    @returns: porcentage of intersection.
    @raises AssertionError: format is not compatible.
    """
    
    # Verify format of bboxes
    assert_input_array(bbox1)
    assert_input_array(bbox2)
    
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    left_x, left_y = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    right_x, right_y = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
     
    area_intersection = (right_x - left_x) * (right_y - left_y)

    if right_x - left_x < 0 or right_y - left_y < 0:
        area_intersection = 0

    intersection_over_union = area_intersection / (area_bbox1 + area_bbox2 - area_intersection + 1e-6)

    return intersection_over_union

