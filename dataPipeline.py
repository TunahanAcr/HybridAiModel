import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class KumasSpektralDataset(Dataset):

    def __init__(self, hibrit_kumas_veriset):
    
        # Veriyi oku
        self.data = pd.read_csv(hibrit_kumas_veriset)

        #18 sütunu al
        X_ham = self.data.iloc[:, :18].values

        # Label sütununu al
        Y_ham = self.data.iloc[:, -1].astype(float).values

        # Veriyi normalize et

        self.x_min = X_ham.min(axis=0)
        self.x_max = X_ham.max(axis=0)

        # Tüm veriyi 0-1 aralığına normalize et
        self.X = (X_ham - self.x_min) / (self.x_max - self.x_min)


        self.Y = Y_ham

        print("Veri seti başarıyla yüklendi ve normalize edildi.")
    


    def __len__(self):
        
        # Toplam kaç satır veri var dataloader bilmeli
        return len(self.data)

    def __getitem__(self, index):
        # Dataloadere index verildiğinde o indexteki veriyi döndür
        # Veriyi tensor formatına çevir çünkü PyTorch ile çalışacağız

        x_tensor = torch.tensor(self.X[index], dtype=torch.float32)
        y_tensor = torch.tensor(self.Y[index], dtype=torch.float32)

        return x_tensor, y_tensor
