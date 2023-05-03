import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import matplotlib.patches as patches
import torch
from torchvision import transforms

def preprocessing_img(img):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                ])
    img = invTrans(img)
    tensor2pil = transforms.ToPILImage()
    img = tensor2pil(img)

    return img

def xyxy2xywh(w, h, bbox: list):
    
    xhalf_side = w * (bbox[2] / 2)
    yhalf_side = h * (bbox[3] / 2)
    left_x, left_y = w * bbox[0] - xhalf_side, h * bbox[1] - yhalf_side

    return [left_x, left_y, 2 * xhalf_side, 2 * yhalf_side]

def plot_bboxes(img: torch.Tensor, bboxes: list, ID: str) -> Image.Image:
    
    """Plot bounding boxes with labels.
        
    :param bboxes: [label, left_x, left_y, right_x, right_y].
    :return img: image with bounding boxes.
    """
    
    img = preprocessing_img(img)

    fig = plt.figure(frameon=False)
    w, h = img.size[:2]
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    
    
    for bbox in bboxes:
        bbox = xyxy2xywh(w, h, bbox) 
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=10, ec=(0.0, 0.0, 1.0), alpha=0.5)

        ax.add_patch(rect)

    fig.savefig(os.getcwd() + f'/ID', bbox_inches='tight', dpi=1)
