import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import Transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


