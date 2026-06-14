import torch.nn as nn

class SpektralModel(nn.Module):
    def __init__(self):
        super (SpektralModel, self).__init__() # Miras alma kuralı ama anlamadım

        # 1. Katman 18 girdi gizli 32 
        # Linear demek: Ağırlıklarla çarpıp bias ekle demek
        self.layer1 = nn.Linear(in_features=18, out_features=32)

        # 2. Katman 32 girdi 1 çıktı 
        self.layer2 = nn.Linear(in_features=32, out_features=1)

        # Filtrelerimiz 
        self.relu = nn.ReLU() # ReLU aktivasyon fonksiyonu
        self.sigmoid = nn.Sigmoid() # Sigmoid aktivasyon fonksiyonu


    def forward(self, x):
        # Tensörün akş yolu 

        x = self.layer1(x) # 1. katman 18 girdiyi 32 nörona çarparak dağıttık 
        x = self.relu(x) # Aktivasyon fonksiyonu Çizgiyi büktük Negatifleri sıfırladık

        x = self.layer2(x) # 2. katman 32 nöronu 1 çıktıya çarptık
        x = self.sigmoid(x) # Aktivasyon fonksiyonu sonuçları 0-1 aralığına sıkıştırdık

        return x