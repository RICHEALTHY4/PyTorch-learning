import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
#download the data


class MyData(Dataset):
    def __init__(self,root_dir,transform = None):
        super().__init__()
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir,'jpg')
        self.transform = transform
        
