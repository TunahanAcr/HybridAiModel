import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv katmanı burada 2d resimlerin 2 boyutlu olmasından kaynaklı in channel 1 ise gri scale resimler için rgb olursa 3 out channel 4 ise 4 farklı kernel kullanarak 4 farklı feature map tablosu üreticez demek stride default 1 gelir kernel kaçar kaçar kaycak onu belirler 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # max pooldan sonra elde ettiğimiz 4 kanallı 3*3 lük tableları 4*3*3 ile tek boyutlu vektöre çeviriyoruz (Flatten) ve 8 nöronumuz var bunu biz belirliyoruz 16 32 falan da olabilirdi
        self.fc1 = nn.Linear(in_features=4*3*3, out_features=8)
        # 8 nörondan çıkan değerleri tek bir nötona sokup final değerimizi elde ediyoruz
        self.fc2 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):

        # Input (1,8,8) 1 gray scale image 8*8 lik resim
        x = self.conv1(x) 
        # Output (4,6,6) 4 kanal 6*6 lık feature tablelar

        x = torch.relu(x)
        x = self.pool(x)
        # Output (4,3,3) 4 kanal feature mapler 3*3 e düşer 

        x = x.view(x.size(0), -1) # .size(0) Tensor'ların ilk boyutu her zaman batch size ı verir .view(batch_size, -1) .view() tensor'un şeklini değiştirir, -1 ise "gerisini sen hesapla" demek: 32 batch'i koru, geri kalan her şeyi (4×3×3) tek bir boyuta düzleştir" diyor. PyTorch otomatik hesaplıyor: 4*3*3=36. Önce: (32, 4, 3, 3)  → 32 görüntü, her biri 4 kanal, 3x3 Sonra: (32, 36)      → 32 görüntü, her biri 36 sayılık düz vektör

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)


        return x

model = SimpleCNN()
print(model)