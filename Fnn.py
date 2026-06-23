import torch.nn as nn

class SpektralModel(nn.Module):
    def __init__(self):
        super().__init__() # Miras alma kuralı bütün nn module özelliklerini alıp kendi layerlarımızı yazıyoruz

        # 1. Katman 18 input 32 nöronun her birine girer
        # 576 weights 32 biasımız olur 
        # nn.Linear demek: Ağırlıklarla çarpıp bias ekle demek
        self.layer1 = nn.Linear(in_features=18, out_features=32)

        # 2. Katmanda 1. Katmandan çıkan 32 değer inputumuz olur 1 nörona girer
        # 32 weights 1 bias olur
        self.layer2 = nn.Linear(in_features=32, out_features=4) # BURADAKİ OUT FEATURE KAÇ SINIFTA TESPİT YAPACAĞIMIZA GÖRE GÜNCELLENCEK

        # Filtrelerimiz 
        self.relu = nn.ReLU() # ReLU aktivasyon fonksiyonu


    def forward(self, x):
        # Tensörün akş yolu 

        x = self.layer1(x) # 1. katman 18 inputu soktuk 
        x = self.relu(x) # 1. Hidden layerdan çıkan 32 değeri relu ya soktuk
        x = self.layer2(x) # Reludan gelen 32 değeri 2. hidden layerımıza soktuk 1 output aldık(logit) 

        return x