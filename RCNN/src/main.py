import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import Tuple
import sys
from torchvision import transforms
import argparse
from tqdm import tqdm
import torch.nn as nn
import selectivesearch
import numpy as np
from dataset import PreprocessingDataset, region_proposal, best_regions, MyDataset

sys.path.append('..')

from utils.visualization import plot_bboxes 
from utils.utils import find_dirs

T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[224, 224]),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

def train(args, dirs):
    
    pre_train_ds = PreprocessingDataset(imgs_dir=dirs['train_images'], annotations_dir=dirs['train_labels'])

    fpath_train = pre_train_ds.imgs_path

    pre_test_ds = PreprocessingDataset(imgs_dir=dirs['test_images'], annotations_dir=dirs['test_labels'])
    
    fpath_test = pre_test_ds.imgs_path

    gtboxes, gtlabels, region_labels, region_bboxes, region_diffs = best_regions(pre_train_ds)

    train_ds = MyDataset(fpath_train, pre_train_ds, gtboxes, gtlabels, region_bboxes, region_labels, region_diffs)

    gtboxes, gtlabels, region_labels, region_bboxes, region_diffs = best_regions(pre_test_ds)

    test_ds = MyDataset(fpath_test, pre_test_ds, gtboxes, gtlabels, region_bboxes, region_labels, region_diffs)

    #img, bboxes, classes, path = pre_train_ds[10]
    #print(bboxes)
    #img = img[:, :, ::-1]
    #cv2.imwrite('teste.jpg', img)
    #test_ds = ImagesDataset(imgs_dir=dirs['test_images'], annotations_dir=dirs['test_labels'])
    
    #train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    #test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, drop_last=True)

def eval():

    pass

def inference(img_path: str):
    

    img = Image.open(img_path).convert('RGB')
    img = T(img)
    # passes takes network and returns image with bouding boxes

    # -----------------
    
    #img, labels = train_ds[10]
    #plot_bboxes(img, labels, 'bbox.jpg')

def get_args():

    parser = argparse.ArgumentParser(description='Region Proposal Convolution Neural Network')
    parser.add_argument('--epochs', required=False, help='sets the number of training epochs')
    parser.add_argument('--data_root_dir', required=True, help='sets the path of the images directory')
    parser.add_argument('--output_model_dir', required=True, help='defines the directory where the network weights will be saved')
    parser.add_argument('--validation_flag', required=False, type=bool, help='flag to use a validationdataset')
    parser.add_argument('--mode', choices=['train', 'eval', 'inference'], default='train', 
            help='choose the mode of use: Training, Evaluation or Inference.')
    parser.add_argument('--image', required=False, help='image path.')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()
    dirs = find_dirs(args)
    
    if args.mode == 'train':
        train(args, dirs)
    elif args.mode == 'eval':
        eval()
    elif args.mode == 'inference':
        inference(args.image)


