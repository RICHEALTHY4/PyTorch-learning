import torch 

distances = torch.tensor([[3.0],[7.0],[12.0]])
weight = 2.0
bias = 8.0

outputs  = weight * distances +bias   #torch.tensor braodcasting
print(outputs)

output1 = torch.tensor([[3.0],[7.0],[12.0]])*torch.tensor([[3.0],[7.0],[12.0]])
print(output1)

character = torch.tensor(
    [
        [3.0,8.0,1.0],
        [7.0,17.0,2.0],
        [12.0,12.0,1.0]
    ]
)
output2 = character * torch.tensor([[1,2,3]]) #1 * first column,2* second column,3 * third column
print(output2)

add_output = torch.tensor([[1,2,3]]) + torch.tensor([[3,2,1]]) #one sample,three character
print(add_output)

add_scalar = torch.tensor([[1,2,3]])+5
print(add_scalar)

#pytorch automatically expand the smaller dimension by repeating values
different_dim = torch.tensor([[1,2,3]]) + torch.tensor([[3],[1]])
print(different_dim) # (1,3) * (2,1)  become (2,3)

different_dim2 = torch.tensor([[1,2,3]]) + torch.tensor([[3]])
print(different_dim2) # (1,3) * (1,1)  become (1,3) 