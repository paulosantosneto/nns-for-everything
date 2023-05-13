import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import selectivesearch
import sys
import cv2
import matplotlib.pyplot as plt

sys.path.append('..')

from utils.visualization import plot_bboxes

def region_proposal(img):
    
    _, regions = selectivesearch.selective_search(img, scale=200, min_size=100) 
    bboxes = []
    
    height, width = img.shape[:2]
    
    for box in regions:
        x, y, w, h = box['rect']
        if w * h > height*0.1 * width*0.1 and w * h < height*0.8 * width*0.8:

            bboxes.append([*box['rect'],])

    img = Image.fromarray(np.uint8(img)).convert('RGB')

    #plot_bboxes(img, bboxes[:2000], 'out.jpg', [])
    
    return bboxes[:2000]





region_proposal(np.asarray(Image.open('teste.jpg')))
