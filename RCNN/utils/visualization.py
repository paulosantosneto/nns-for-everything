import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import matplotlib.patches as patches
import torch
from torchvision import transforms
import cv2
import pyshine as ps

def preprocessing_img(img: list):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                ])
    img = invTrans(img)
    tensor2pil = transforms.ToPILImage()
    img = tensor2pil(img)

    return img

def rescaling(w: float, h: float, bbox: list):

    bbox[0] = w * bbox[0]
    bbox[1] = h * bbox[1]
    bbox[2] = w * bbox[2]
    bbox[3] = h * bbox[3]

    return bbox

def xywh2xyxy(bbox: list):
    
    xhalf_side = bbox[2] / 2
    yhalf_side = bbox[3] / 2
    left_x, left_y = bbox[0] - xhalf_side, bbox[1] - yhalf_side

    return [left_x, left_y, left_x + bbox[2], left_y + bbox[3]]

def complete_array(ar: list, length: int, complete: any):
    
    new_ar = []

    for i in range(length):
        if i < len(ar):
            new_ar.append(ar[i])
        else:
            new_ar.append(complete)

    return new_ar

def plot_bboxes(img: any, bboxes: list, ID: str, labels: list) -> list:
    
    """Plot bounding boxes with labels.
        
    :param bboxes: [label, left_x, left_y, right_x, right_y].
    :return img: image with bounding boxes.
    """

    img = img[:, :, ::-1].copy()
     
    labels = complete_array(labels, len(bboxes), '')
    print(img.shape)
    for bbox, label in zip(bboxes, labels):
        lx, ly, rx, ry = [int(value) for value in bbox]
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (103, 146, 137), 2)
        #img = cv2.rectangle(img, (int(bbox[0])-2, int(bbox[1])-40), (int(bbox[0])+50, int(bbox[1])-2), (103, 146, 137), -1)
        img = cv2.putText(img, label, (int(bbox[0])-5, int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (103, 146, 137), 1, cv2.LINE_AA) 
        #img = ps.putBText(img, label, text_offset_x=lx-2, text_offset_y=ly-40, vspace=10, hspace=10, font_scale=1.0, background_RGB=(103, 146, 137), text_RGB=(0, 0, 0))
    cv2.imwrite(os.getcwd() + f'/{ID}', img)

