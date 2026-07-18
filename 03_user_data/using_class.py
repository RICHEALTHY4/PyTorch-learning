class Cat():
    def __init__(self,name,age):
        self.name = name 
        self.age = age
    def meow(self):
        print(f"my name is {self.name}, I am {self.age} years old. ")

cat1 = Cat("Tom",3)
cat1.meow()
cat2 = Cat("Kitty",5)
cat2.meow()

class Calculator():

    def add(self,a,b):
        return a+b
    
calc = Calculator()
print(calc.add(3,5))

import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],dtype = torch.float32)
label = torch.tensor([[5],[8],[10.9],[14],[16.9],[20],[23],[25.9],[30],[34]])

class Mydata(Dataset):
    def __init__(self,data,label):
        super().__init__()
        self.data = data
        self.label = label
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        return (self.data[index]) ,(self.label[index])
        
dataset = Mydata(data,label)
print(len(dataset))
print(dataset[0])
print(dataset[5])

for sample in dataset:
    print(sample)

dataloader = DataLoader(
    dataset,  # where data from
    batch_size= 2, # how much data one time
    shuffle  = True
)

for x,y in dataloader:
    print(x)
    print(y)