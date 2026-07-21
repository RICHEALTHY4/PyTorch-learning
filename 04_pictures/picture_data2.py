
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
test_loader = DataLoader(test_dataset,batch_size = 32,shuffle = False)

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
        self.fc1 = nn.Linear(64*56*56,512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,102)


    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x= self.dropout(x)
        x = self.fc2(x)
        return x
    
model = SimpleCNN()
loss_functioin = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr= 0.0005,weight_decay = 0.0005)
total_loss = 0
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
best_accuracy = 0



for epoch in range(100):
    model.train()
    train_loss=0
    correct = 0
    total = 0
    for img_batch,label_batch in train_loader:
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        output = model(img_batch)
        train_loss_value = loss_functioin(output,label_batch)
        train_loss_value.backward()
        optimizer.step()
        prediction = output.argmax(dim = 1)
        correct += (prediction == label_batch).sum().item()
        total += label_batch.size(0)
        train_loss += train_loss_value.item()

    accuracy = correct / total 
    print(f"train epoch {epoch} accuracy{accuracy}")

    if epoch % 50 ==0 :
        print(f"total_loss: {train_loss}")
        
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for img_val,label_val in val_loader:
            img_val = img_val.to(device)
            label_val = label_val.to(device)

            val_output = model(img_batch)
            val_loss_value = loss_functioin(val_output,label_batch)
            val_prediction = val_output.argmax(dim = 1)
            val_correct += (prediction == label_batch).sum().item()
            val_total += label_batch.size(0) 
        val_accuracy = val_correct / val_total 
        print(f"validation epoch {epoch} accuracy{accuracy}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                "best_model.pth"
            )


def test(model ,test_loader):
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data,label in test_loader:
            data = data.to(device)
            label = label.to(device)

            test_out = model(data)
            prediction = torch.argmax(test_out,dim = 1)
            correct += (prediction == label).sum().item()  #use item() to vonvert to int
            total += label.size(0)

        accuracy = correct / total
        print(f"test accuracy: {accuracy:.4f}")

test(model,test_loader)

