"""
print("this is a test!")
import torch
a = torch.tensor(2.0)
b = torch.tensor(3.0)
result = a + b
print(result)
"""

import torch
import torch.nn as nn
import torch.optim as optim

#define the model
model = nn.Sequential( #one input ,one output,linear transformation
    nn.Linear(1,1)
)  
#define the loss fuction and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.001)

distances = torch.tensor([[3.0],[7.0],[12.0],[18.0],[22.0],[28.0]])
print(distances.shape)
times = torch.tensor([[14.8],[24.1],[36.5],[51.2],[60.8],[75.3]])
print(times.shape)
distances = distances / 30.0
times = times / 80.0


#training loop
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(distances)
    loss = loss_function(outputs,times)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    test_distance = torch.tensor([[25.0/30]])
    predicted_time = model(test_distance)*80
    print(f"Predicted time for 25 miles:{predicted_time.item():.1f} minutes")




