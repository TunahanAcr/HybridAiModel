import torch
import torch.nn as nn

dataset = [
    (6.0, 8.0, 9.5),
    (1.0, 9.0, 3.0),
    (4.0, 6.0, 6.5),
    (10.0, 1.0, 4.0)
]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden = nn.Linear(2,2)
        self.output = nn.Linear(2,1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))

        return self.output(x)



model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range (100):
    for x1, x2, target in dataset:
        x = torch.tensor([x1, x2])
        y = torch.tensor([target])

        prediction = model(x)
        loss = loss_fn(prediction,y)

        # PyTorch'ta gradyanlar otomatik biriken bir yapıda tutuluyor loss.backward() her çağrıldığında eskisinin üzerine ekliyor. zero_grad() bunu sıfırlar, yoksa eski batch'in gradyanları yeni batch'inkiyle toplanır, yanlış güncelleme olur.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

x = torch.tensor([20.0, 0.0])
print(model(x))