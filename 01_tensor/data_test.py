import torch 
"""
distances = torch.tensor([[3.0],[7.0],[12.0],[18.0],[22.0],[28.0]])
print(distances.shape)  #torch.size([6,1]),6 sample, 1 feature per sample


int_tensor = torch.tensor([1,2,3],dtype = torch.int64)
float_tensor = torch.tensor([1.0,2.0,3.0],dtype=torch.float32)
mixed_tensor = float_tensor + int_tensor
print(f"mixed type:{mixed_tensor.dtype}") #mixed type:torch.float32

zeros = torch.tensor(3,3 )
ones = torch.ones(2,4)
random = torch.rand(5,5)

single_distance = torch.tensor(25.0)
print(single_distance)
#print(f"single_distance.dtype {single_distance.dtype}")
with_batch = single_distance.unsqueeze(0)  #added batch dim
print(with_batch)
print(with_batch.shape)
ready_for_model = with_batch.unsqueeze(1)   #added charater dim
print(ready_for_model)
print(ready_for_model.shape)

squeezed = ready_for_model.squeeze()
print(squeezed)
print(squeezed.shape)
"""

#indexing and sclicing
prediction = torch.tensor([[14.9],[24.1],[35.6],[45.2]])
first = prediction[0]
print(first)

first_three = prediction[:3]
print(first_three)

#use .item() to get a python number，only for tensor with one item
value = prediction[0].item()
print(value)

#multiple fearures per sample
data = torch.tensor(
    [[3.0,8.0,1.0],  #disatance hour weather
    [7.0,17.0,2.0],
    [12.0,12.0,1.0]],
)
distances = data[:, 1]
print(distances.size())
print(distances)