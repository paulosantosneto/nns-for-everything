import torch
from torch.utils.data import Dataset
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

sys.path.append('..')

from utils.visualization import plot_bboxes 
from utils.utils import find_dirs

T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

class ImagesDataset(Dataset):

    def __init__(self, imgs_dir: str, annotations_dir: str):
        self.imgs_id = glob(imgs_dir+'/*.jpg')
        self.imgs_path = glob(os.path.join(imgs_dir, '*.jpg'))
        self.annotations_path = glob(os.path.join(annotations_dir, '*.txt')) 
        self.labels = self.load_labels()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_labels(self) -> dict:
                
        labels = {}
        
        for annot_path, ID in zip(self.annotations_path, self.imgs_id):
            labels[ID] = [] 
            with open(annot_path) as f:
                for line in f.readlines():
                    bboxes = list(map(float, line.split()))
                    bboxes.append(bboxes[0])
                    labels[ID].append(bboxes[1:])
        
        return labels
    
    def __getitem__(self, ix) -> Tuple[list, dict]:
                
        pil_img = Image.open(self.imgs_path[ix]).convert('RGB')
        img = T(pil_img)
        img_labels = self.labels[self.imgs_id[ix]]
        
        return img.float().to(self.device), img_labels
    
    def __len__(self):
        return len(self.imgs_path)

class RCNN(nn.Module):

    def __init__(self, epochs: int=5):
        self.epochs = epochs

def train():

    pass

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
        train(args)
    elif args.mode == 'eval':
        eval()
    elif args.mode == 'inference':
        inference(args.image)

    train_ds = ImagesDataset(imgs_dir=dirs['train_images'], annotations_dir=dirs['train_labels'])
    test_ds = ImagesDataset(imgs_dir=dirs['test_images'], annotations_dir=dirs['test_labels'])

