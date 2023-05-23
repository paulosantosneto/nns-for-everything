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

def train(args, dirs):
    
    # preprocessing train, test and validation datasets

    pre_train_ds = PreprocessingDataset(imgs_dir=dirs['train_images'], annotations_dir=dirs['train_labels'])
    pre_test_ds = PreprocessingDataset(imgs_dir=dirs['test_images'], annotations_dir=dirs['test_labels'])

    fpath_train, fpath_test = pre_train_ds.imgs_path, pre_test_ds.imgs_path

    if args.validation_flag:
        pre_val_ds = PreprocessingDataset(imgs_dir=dirs['val_images'], annotations_dir=dirs['val_labels'])
        fpath_val = pre_val_ds.imgs_path

    label2idx, idx2label = pre_train_ds.label2idx, pre_train_ds.idx2label
    
    # Transform dataset to tensors units

    train_ds = MyDataset(fpath_train, *best_regions(pre_train_ds, mode='train'), label2idx, idx2label)
    
    # DataLoader for train, test and validation

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=train_ds.collate_fn, drop_last=False)
    
    for i, batch in enumerate(iter(train_dl)):
        print(batch)
        break
    #cv2.imwrite('testecrop.jpg', crop_regions[0])
    """    
    gtboxes, gtlabels, region_labels, region_bboxes, region_diffs = best_regions(pre_test_ds, mode='test')

    test_ds = MyDataset(fpath_test,  gtboxes, gtlabels, region_bboxes, region_labels, region_diffs)
    """
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


