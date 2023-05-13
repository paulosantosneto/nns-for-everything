from glob import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
import sys
import os
import selectivesearch
import re

sys.path.append('..')

from utils.utils import IOU
from utils.visualization import rescaling, xywh2xyxy, plot_bboxes

def region_proposal(img):

    _, regions = selectivesearch.selective_search(img, scale=100, min_size=100)
    bboxes = []
    height, width = img.shape[:2]

    for box in regions:
        
        x, y, w, h = box['rect']
        if w * h > height*0.1 * width*0.1 and w * h < height*0.8 * width*0.8:
            bboxes.append([x, y, x+w, y+h])

    return bboxes

def fit_with_gt(img, regions: list, gtboxes: list, gtlabels):
    
    labels = []
    best_regions_iou = []
    diffs = []

    for i, box in enumerate(gtboxes):
        clean_regions = [(region, IOU(region, box)) for region in regions if IOU(region, box) > 0.2]
        higher_iou_bbox = sorted(clean_regions, key=lambda x: x[1], reverse=True)
            
        best_regions_iou.append(higher_iou_bbox[0][0])
        gt_left_x, gt_left_y, gt_right_x, gt_right_y = box
        region_left_x, region_left_y, region_right_x, region_right_y = higher_iou_bbox[0][0]
        diffs.append([gt_left_x - region_left_x, gt_left_y - region_left_y /  
                    gt_right_x - region_right_x, gt_right_y - region_right_y])

        if len(higher_iou_bbox) > 0:
            labels.append(gtlabels[i])
        else:
            labels.append('background')

    return best_regions_iou, labels, diffs

def best_regions(data):
    
    regions_labels = {}
    regions_bboxes = {}
    regions_diffs = {}

    for ix, ds in enumerate(iter(data)):
        
        print(ix)        
        img, bboxes, classes, path, maplabel = ds
        labels = [maplabel[label] for label in classes]
        plot_bboxes(img.copy(), bboxes, 'ground_truth.jpg', labels)
        regions = region_proposal(img.copy())
        plot_bboxes(img.copy(), regions, 'all_regions.jpg', [])
        best_region, rlabels, diffs = fit_with_gt(img.copy(), regions, bboxes, labels)
        plot_bboxes(img.copy(), best_region, 'best_regions.jpg', rlabels)
        
        if len(best_region) > 0:
            
            regions_labels[path] = rlabels
            regions_bboxes[path] = best_region
            regions_diffs[path] = diffs

        break

    return bboxes, classes, regions_labels, regions_bboxes, regions_diffs

class PreprocessingDataset(Dataset):

    def __init__(self, imgs_dir: str, annotations_dir: str):
        
        self.imgs_id = glob(imgs_dir+'/*.jpg')
        self.imgs_path = glob(os.path.join(imgs_dir, '*.jpg'))
        self.annotations_path = glob(os.path.join(annotations_dir, '*.txt')) 
        self.annotations_path = [f for f in self.annotations_path if f != os.path.join(annotations_dir, 'classes.txt')]
        self.labels = self.load_labels()
        self.classes = self.load_classes(annotations_dir)
        
    def load_classes(self, annotations_dir: str) -> dict:
        
        try:
            classes = {} 
            with open(os.path.join(annotations_dir, 'classes.txt')) as f:
                for ix, name in enumerate(f.readlines()):
                    classes[int(ix)] = re.sub("\n", "", name)

            return classes
        except:
            raise 'The class file has some problme in its structure.'

    def load_labels(self) -> dict:
                
        labels = {}
        
        for annot_path, img_id in zip(self.annotations_path, self.imgs_id):
            labels[img_id] = {'bbox': [], 'class': []}
            with open(annot_path) as f:
                for line in f.readlines():
                    bboxes = list(map(float, line.split()))
                    labels[img_id]['bbox'].append(bboxes[1:])
                    labels[img_id]['class'].append(bboxes[0])
            
        return labels
    
    def __getitem__(self, ix) -> Tuple[list, list, list, str]:
                
        img = cv2.imread(self.imgs_path[ix])[:, :, ::-1].copy()
        bboxes = self.labels[self.imgs_id[ix]]['bbox']
        classes = self.labels[self.imgs_id[ix]]['class']
        
        h, w = img.shape[:2]
        
        bboxes = [rescaling(w, h, bbox) for bbox in bboxes]
        bboxes = [xywh2xyxy(bbox) for bbox in bboxes]
    
        return img, bboxes, classes, self.imgs_path[ix], self.classes

    def __len__(self):

        return len(self.imgs_path)

class MyDataset(Dataset):

    def __init__(self, fpath, pre_train_ds, gtboxes, gtlabels, region_bboxes, region_labels, region_diffs):
        self.pre_train_ds = pre_train_ds
        self.gtboxes = gtboxes
        self.gtlabels = gtlabels
        self.region_bboxes = region_bboxes
        self.region_labels = region_labels
        self.region_diffs = region_diffs
        self.fpath = fpath

        print('Ground Truth bboxes:', gtboxes)
        print('Ground Truth labels:', gtlabels)
        print('Regions bboxes:', region_bboxes)
        print('Regions labels:', region_labels)
        print('Regions diffs:', region_diffs)

    def collate_fn(self):

        pass

    def __getitem__(self, ix):

        img = cv2.imread(self.fpath[ix])[:, :, ::-1]
        region_bboxes = self.region_bboxes 
        crop_regions = [img[lx:rx, ly:ry] for (lx, ly, rx, ry) in self.region_bboxes[ix]]
        gtbboxes = self.gtbboxes[ix]
        region_labels = self.region_labels[ix]
        diffs = self.region_diffs[ix]

        return img, region_bboxes, gtbboxes, region_labels, crop_regions, diffs

    def __len__(self):
        
        return len(self.pre_train_ds)
