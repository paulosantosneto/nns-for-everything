import numpy as np
from typing import Optional
import torch
import os

def assert_input_array(bbox):
    
    try:
        assert len(bbox) == 4, bbox.dtype == np.float32
    except AssertionError as e:
        e.args += ('length isn ot compatible', 'type of data is not compatible')

def find_dirs(args: dict):
    
    ROOT = args.data_root_dir
    paths = {}

    if not os.path.exists(ROOT + '/train'):
        raise Exception('Training directory not found!')
    if not os.path.exists(ROOT + '/test'):
        raise Exception('Test directory not found!')
    if not os.path.exists(ROOT + '/val') and args.validation_flag:
        raise Exception('Validation directory not found!')

    paths['train'] = ROOT + '/train/'
    paths['test'] = ROOT + '/val/'
    paths['val'] = ROOT + '/test/'
    paths['train_images'] = ROOT + '/train/images/'
    paths['train_labels'] = ROOT + '/train/labels/'
    paths['test_images'] = ROOT + '/test/images/'
    paths['test_labels'] = ROOT + '/test/labels/'

    return paths

def IOU(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate intersection over union between two bounding boxes.

    :param bbox1: bounding boxes 1 [left_x, left_y, right_x, right_y].
    :param bbox2: bounding boxes 2 [left_x, left_y, right_x, right_y].
    :returns: porcentage of intersection.
    :raises AssertionError: format is not compatible.
    """
    
    assert type(bbox1) == list
    assert type(bbox2) == list

    bbox1_area = abs(bbox1[0] - bbox1[2]) * abs(bbox1[1] - bbox1[3])
    bbox2_area = abs(bbox2[0] - bbox2[2]) * abs(bbox2[1] - bbox2[3])
    
    bbox_intersection = (max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3]))
    intersection_area = (bbox_intersection[2] - bbox_intersection[0]) * (bbox_intersection[3] - bbox_intersection[1])
    
    if (bbox_intersection[2] - bbox_intersection[0]) < 0 or (bbox_intersection[3] - bbox_intersection[1]) < 0:
        intersection_area = 0

    iou_value = intersection_area / (bbox1_area + bbox2_area - intersection_area + 1e-6)

    return iou_value
