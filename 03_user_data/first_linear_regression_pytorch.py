import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

x = torch.tensor([[1],[2],[3],[4],[5],[6]],dtype = torch.float32)
y = torch.tensor([[4],[5.9],[7.9],[10.1],[12.2],[14]],dtype = torch.float32)

class MyData(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return(len(self.x))
    def __getitem__(self,index):
        return self.x[index],self.y[index]

dataset = MyData(x,y)
dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True
)

x_test = torch.tensor([[10],[20]],dtype = torch.float32)
y_test = torch.tensor([[22],[42]],dtype = torch.float32)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,1)

    def forward(self,x):
        x = self.l1(x)
        return x

model = MyModel()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)

for epoch in range(500):
    model.train()
    total_loss = 0.0

    for x_batch,y_batch in dataloader:
        #print(f"x_shape {x.shape},y_shape{y.shape}")
        optimizer.zero_grad()
        outputs =model(x_batch)
        loss_value = loss_function(outputs,y_batch)
        #print(loss_value)
        '''
        print("before:")  # to check the weight and the bias
        print(model.l1.weight)
        print(model.l1.bias)
        '''
        loss_value.backward()
        optimizer.step()
        '''
        print("after:")
        print(model.l1.weight)
        print(model.l1.bias)
        '''
        total_loss += loss_value.item()
        #print(f"loss_value.item() type {type(loss_value.item())}")
    
    if epoch %50 ==0:
        print(f"epoch: {epoch},total_loss :{total_loss}")

model.eval()
with torch.no_grad():
    prediction = model(x_test)
    print(f"eval mode: {prediction}")



