from torch.utils.data import Dataset
import os
import scipy
from PIL import Image


class RobustFlowerDataset(Dataset):
    def __init__(self,root_dir,transform = None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir,'jpg')
        self.transform = transform

        # load labels
        labels_mat = scipy.io.loadmat(os.path.join(root_dir,'imagelabels.mat'))
        self.labels = labels_mat['labels'][0]-1

        #new:keep track of any errors we encounter
        self.erro_log = []
    
    def __getitem__(self,idx):
        try:
            #normal loading
            img_name = f'imag_{idx+1:05d}.jpg'
            img_path = os.path.join(self.img_dir,img_name)
            image = Image.open(img_path)
        
            #check for corruption
            image.veryfy()   #verify() close the file
            image = Image.open(img_path) #reopen for transform use

            #skip tiny images
            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError(f"Image too small:{image.size}")
            #convert grascale to RGB
            if image.mode != 'RGB':
                image = image.conver('RGB')
        except Exception as e:
            #log the issue instead of crashing
            self.error_log.append({
                'index':idx,
                'error':str(e),
                'path':img_path if 'img_path' in locals() else 'unkown'
            })
            print(f"warning:skipping corrupted image{idx}:{e}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

