from glob import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
import sys
import os
import selectivesearch
import re
from torchvision import transforms
import numpy as np
import torch

sys.path.append('..')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from utils.utils import IOU
from utils.visualization import rescaling, xywh2xyxy, plot_bboxes

T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[224, 224]),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

def tensor2numpy(img):
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]), ])
    img = invTrans(img)
    img = (img.numpy().squeeze().transpose((1, 2, 0)) * 255.0).astype(np.uint8)

    return img


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
        clean_regions = [(region, IOU(region, box)) for region in regions if IOU(region, box) > 0.3]
        higher_iou_bbox = sorted(clean_regions, key=lambda x: x[1], reverse=True)
        
        if len(higher_iou_bbox) > 0:

            best_regions_iou.append(higher_iou_bbox[0][0])
            gt_left_x, gt_left_y, gt_right_x, gt_right_y = box
            region_left_x, region_left_y, region_right_x, region_right_y = higher_iou_bbox[0][0]
            diffs.append([gt_left_x - region_left_x, gt_left_y - region_left_y, gt_right_x - region_right_x, gt_right_y - region_right_y])

            labels.append(gtlabels[i])
        else:
            labels.append('background')

    return best_regions_iou, labels, diffs

def best_regions(data, mode='train'):
    
    regions_labels = {}
    regions_bboxes = {}
    regions_diffs = {}
    gtbboxes = {}
    gtclasses = {}

    for ix, ds in enumerate(iter(data)):
        
        if ix == 0:
            img, bboxes, classes, path, idx2label, label2idx = ds
            labels = [idx2label[idx] for idx in classes]
            plot_bboxes(img.copy(), bboxes, f'{mode}_ground_truth_{ix}.jpg', labels)
            regions = region_proposal(img.copy())
            plot_bboxes(img.copy(), regions, f'{mode}_all_regions_{ix}.jpg', [])
            best_region, rlabels, diffs = fit_with_gt(img.copy(), regions, bboxes, labels)
            plot_bboxes(img.copy(), best_region, f'{mode}_best_regions_{ix}.jpg', rlabels)
        
            if len(best_region) > 0:
            
                regions_labels[path] = rlabels
                regions_bboxes[path] = best_region
                regions_diffs[path] = diffs
                gtbboxes[path] = bboxes
                gtclasses[path] = [idx2label[idx] for idx in classes]

            return gtbboxes, gtclasses, regions_bboxes, regions_labels, regions_diffs

class PreprocessingDataset(Dataset):

    def __init__(self, imgs_dir: str, annotations_dir: str):
         
        self.imgs_id = glob(imgs_dir+'/*.jpg')
        self.imgs_path = glob(os.path.join(imgs_dir, '*.jpg'))
        self.annotations_path = glob(os.path.join(annotations_dir, '*.txt')) 
        self.annotations_path = [f for f in self.annotations_path if f != os.path.join(annotations_dir, 'classes.txt')]
        self.labels = self.load_labels()
        self.idx2label, self.label2idx = self.load_classes(annotations_dir)
        
    def load_classes(self, annotations_dir: str) -> dict:
        
        try:
            label2idx = {}
            idx2label = {}
            print(os.path.join(annotations_dir, 'classes.txt'))
            with open(os.path.join(annotations_dir, 'classes.txt')) as f:
                for ix, name in enumerate(f.readlines()):
                    label = re.sub("\n", "", name)
                    label2idx[label] = int(ix)
                    idx2label[int(ix)] = label

            backgroundidx = len(label2idx.keys())
            label2idx['background'] = backgroundidx
            idx2label[backgroundidx] = 'background'
            
            print(label2idx)
            print(idx2label)

            return idx2label, label2idx
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
    
        return img, bboxes, classes, self.imgs_path[ix], self.idx2label, self.label2idx

    def __len__(self):

        return len(self.imgs_path)

class MyDataset(Dataset):

    def __init__(self, fpath,  gtboxes, gtlabels, region_bboxes, region_labels, region_diffs, label2idx, idx2label):

        self.gtboxes = gtboxes
        self.gtlabels = gtlabels
        self.region_bboxes = region_bboxes
        self.region_labels = region_labels
        self.region_diffs = region_diffs
        self.fpath = fpath
        self.label2idx = label2idx
        self.idx2labels = idx2label

        print('Ground Truth bboxes:', gtboxes)
        print('Ground Truth labels:', gtlabels)
        print('Regions bboxes:', region_bboxes)
        print('Regions labels:', region_labels)
        print('Regions diffs:', region_diffs)

    def collate_fn(self, batch):
        
        x, labels, diffs = [], [], []

        for i in range(len(batch)):
            img, rboxes, gtboxes, rlabels, rcrops, diffs, gtlabels = batch[i]
            #cv2.imwrite('teste_2.jpg', rcrops[0])    
            rcrops = [cv2.cvtColor(rcrop, cv2.COLOR_BGR2RGB) for rcrop in rcrops]
            rcrops = [T(rcrop) for rcrop in rcrops]
            #H, W = rcrops[0].shape[:2] 
            #img_teste = tensor2numpy(rcrops[0])

            #cv2.imwrite('teste.jpg', img_teste)
            
            x.extend(rcrops)

            labels.extend([self.label2idx[label] for label in rlabels])

            diffs.extend(diffs)

        print(x)
        x = torch.cat(x).to(DEVICE)
        
        return x, torch.Tensor(labels).long().to(DEVICE), torch.Tensor(diffs).float().to(DEVICE)      

    def __getitem__(self, ix):
    

        ix = self.fpath[ix]

        img = cv2.imread(ix)[:, :, ::-1]
        region_bboxes = self.region_bboxes 
        crop_regions = [img[ly:ry, lx:rx]  for (lx, ly, rx, ry) in self.region_bboxes[ix]]
        gtbboxes = self.gtboxes[ix]
        region_labels = self.region_labels[ix]
        diffs = self.region_diffs[ix]
        gtlabels = self.gtlabels[ix] 

        return img, region_bboxes, gtbboxes, region_labels, crop_regions, diffs, gtlabels

    def __len__(self):
        
        return len(self.fpath)
