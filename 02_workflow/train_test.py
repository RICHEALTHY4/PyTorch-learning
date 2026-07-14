"""loss = loss_function(outputs,targets)
loss.backward()  #backward only calculate gradients 
optimizer.step() #update the weights


# to choose a device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#move parameters to the selected device
model = MyModel().to(device)


#training loop,move each batch of data to the selected device
for inputs,targets in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)

#check the device
print(inputs.device)
print(next(model.parameters()).device)

#pay attention
x.to(device)  #this  actually creates a new device
x = x.to(device)


#put together
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

for inputs, targets in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs,targets)
    loss.backward()
    optimizer.step()

    #for minist 
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    #image data processing
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ]
    )
"""
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
transform = transforms.ToTensor()
#Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root = "./data",train = True,download = True,transform = transform)
test_dataset = torchvision.datasets.MNIST(root="./data",train = False,download = True,transform = transform)

#create data loaders 
train_loader = DataLoader(train_dataset,batch_size = 64,shuffle = True) #check the images in a different random order
test_loader = DataLoader(test_dataset,batch_size = 1000,shuffle = False)

    #create the Neural Network
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
    
#check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

#Initialize model and move to device
model = MNISTClassifier().to(device)

#Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

#training
def train_epoch(model,train_loader,loo_function,optimizer,device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()

        #track progress
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        #print every 100 batches
        if batch_idx % 100 ==0 and batch_idx >0:
            avg_loss = running_loss /100
            accuracy = 100. * correct/total 
            print(f'[{batch_idx *64}/{60000}]'
                    f'Loss:{avg_loss:.3f} | Accuracy:{accuracy:.1f}%')
            running_loss = 0.0

def evaluate(model,test_loader,device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs,targets in test_loader:
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.* correct/total
#trainning loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f'\nEpoch :{epoch +1}')
    train_epoch(model,train_loader,loss_function,optimizer,device)
    accurracy = evaluate(model,test_loader,device)
    print(f'Test Accuracy:{accurracy:.2f}%')