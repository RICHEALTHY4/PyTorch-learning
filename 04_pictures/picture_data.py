import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import os
import scipy
import time
import torch.optim as optim
from torchvision.datasets import Flowers102

'''
class FlowerDataset(Dataset):
    def __init__(self,root_dir,transform = None):
        super().__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir,'jpg')
        labels_mat = scipy.io.loadmat(os.path.join(root_dir,'imagelabels.mat'))
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        #if self.transform:
            #image=self.transform(image)
    # idx on dataset begin with 0
        self.labels = labels_mat['labels'][0] - 1
        self.error_log = []

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        try:

            img_name = f"image_{idx + 1:.5d}.jpg"
            img_path = os.path.join(self.image_dir,img_name)
            image = Image.open(img_path)

            image.verify()  # to check file
            image = Image.open(img_path) #reopen for other usage

            label = self.labels[idx]

            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError(f"Image too small:{image.size}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            #log the issue instead of crashing
            self.error_log.append(
                {
                    'index':idx,
                    'error':str(e),
                    'path':img_path if 'img_path' in locals() else 'unkown'
                }
            )
            print(f"warning:skipping corrupteed image{idx}:{e}")
            next_idx = (idx + 1)%len(self)
            return self.__getitem__(next_idx)
        if self.transform:
            image = self.transform(image)
        return image,label


#splitting dataset
from torch.utils.data import random_split
train_size = int(0.7* len(dataset))
val_size = int(0.15 * len(dataset))
test_size = int(0.15 * len(dataset))

train_dataset,val_dataset,test_dataset = random_split(dataset,[train_size,val_size,test_size])
print(f"Training :{len(train_dataset)}")
print(f"Validation:{len(val_dataset)}")
print(f"Test:{len(test_dataset)}")
'''  
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean= [0.485,0.456,0.406],std = [0.229,0.224,0.225])
])
  
train_dataset = Flowers102(
    root="./flower_data",
    split="train",
    download=True,
    transform=transform
)

val_dataset = Flowers102(
    root="./flower_data",
    split="val",
    download=True,
    transform=transform
)

test_dataset = Flowers102(
    root="./flower_data",
    split="test",
    download=True,
    transform=transform
)

#using one picture to make sure data size is resized as planed
img, _ = train_dataset[0]
print(f"size of image{img.size}")
print(f"shape of image{img.shape}")

train_loader = DataLoader(train_dataset,batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset,batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset,batch_size = 32,shuffle = True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32,kernel_size = 3,padding =1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,padding =1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2)       
    
        #flatten layer no parameters,just reshaping
        self.flatten = nn.Flatten()

        #fully connected layer
        self.fc1 = nn.Linear(64*56*56,256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256,102)


    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x= self.dropout(x)
        x = self.fc2(x)
        return x
    
model = SimpleCNN()
loss_functioin = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr= 0.0005,weight_decay = 0.0005)
total_loss = 0

for epoch in range(100):
    model.train()
    total_loss=0
    for img_batch,label_batch in train_loader:
        optimizer.zero_grad()
        output = model(img_batch)
        loss_value = loss_functioin(output,label_batch)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()

    if epoch % 50 ==0 :
        print(f"total_loss: {total_loss}")
        

model.eval()
for epoch in range(50):
    with torch.no_grad():
        for img_batch,label_batch in val_loader:
            output = model(img_batch)
            loss_value = loss_functioin(output,label_batch)

for img_batch,label_batch in test_loader:
    output = model(img_batch)
    preidiction = torch.argmax(output,dim = 1)
