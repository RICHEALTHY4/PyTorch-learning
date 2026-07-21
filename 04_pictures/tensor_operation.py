import torch
x = torch.tensor([
    [1,2,3],
    [4,5,6]
])

print(x.shape) #torch.Size(2,3)
print(x.size())#torch.Size(2,3)
print(x.size(0)) #2
print(x.size(1)) #3
print(x.dim()) #2  two sample,two dimention 


#dim = 0,zongxiang
#dim = 1 ,hengxiang
print(x.sum(dim = 0))  #tensor([5,7,9])
print(x.sum(dim = 1)) #tensor([6,15])

output = torch.tensor([
    [2.1, 0.5, 3.2],
    [0.3, 5.1, 1.0],
    [8.0, 0.2, 0.1]
])

print(output.argmax(dim = 1)) # output = tensor([[3.2] , [5.1], [8.0]])
'''
import torch.nn as nn
loss_function = nn.CrossEntropyLoss()
loss = loss_function(output)  #loss is tensor
print(loss.item())  # float
'''
x = torch.tensor([1,2,3])
print(x.shape)
x1 = x.unsqueeze(0)  
print(x1)           #tensor([[1, 2, 3]])
print(x1.shape)
x2 = x.unsqueeze(1)
print(x2)
print(x2.shape)   
'''              tensor([[1],
                        [2],
                        [3]])'''

'''You can insert a new dimension into an n-dimensional tensor 
at any position from 0 to n.'''
xx = torch.randn(2,3)
xx1 = xx.unsqueeze(2)
print(xx1)
print(xx1.shape)

y = torch.tensor([[4,5,6]])
y1 = y.squeeze(0)
print(y)     #tensor([[4, 5, 6]])
print(y1)    #tensor([4, 5, 6])

z = torch.randn(1,3,1)
print(z.shape)
z1 = z.squeeze()  #only delete dimention = 1; torch.Size([3])
print(z1.shape)
z2 = z.squeeze(2) #torch.Size([1, 3])
print(z2.shape)

w = torch.arange(12)
print(w,w.shape)
print(w.reshape(3,4)) #from left to right ;from top to bottom
w1 = w.reshape(2,-1)
print(w1.shape)

#change dimention sequence
r = torch.randn(2,3,4)
print(f"r shape:{r.shape}")
r1 = r.permute(1,0,2)         #r shape:torch.Size([2, 3, 4])
print(f"r1 shape:{r1.shape}") #r1 shape:torch.Size([3, 2, 4])
#rgb picture [height,width,channel]
#torch rgb   [channel,height,width]
pic = torch.randn(224,224,3)
pic_torch = pic.permute(2,0,1) #pic_torch shape torch.Size([3, 224, 224])
print(f"pic_torch shape {pic_torch.shape}")

#Concatenate along the specified dimension; other dimensions must be identical
a = torch.tensor([[1,2],
                  [3,4]])
b = torch.tensor([[5,6],
                  [7,8]])
c = torch.cat([a,b],dim = 0)
print(c)
d = torch.cat([a,b],dim = 1)
print(d)
'''
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
'''


#increase a new dimention
a = torch.tensor([1,2])
b = torch.tensor([3,4])
c = torch.stack([a,b])   #default dim = 0
print(f"stack[a,b] shape{c.shape}")
print(c)
'''
tensor([[1, 2],
        [3, 4]])
'''
d = torch.stack([a,b],dim = 1)
print(f"stack[a,b] ,dim = 1,{d.shape}")
print(d)
'''
tensor([[1, 3],
        [2, 4]])
'''