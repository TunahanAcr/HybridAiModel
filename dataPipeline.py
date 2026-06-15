import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

BLACK_REF = np.array([100, 105, 98, 110, 100, 95, 102, 101, 99, 
    100, 105, 98, 110, 100, 95, 102, 101, 99])  # Siyah referans

WHITE_REF = np.array([1100, 1150, 1080, 1120, 1100, 1090, 1110, 1105, 1095, 1100, 1150, 1080, 1120, 1100, 1090, 1110, 1105, 1095])  # Beyaz referans

def calibrate_data(ham_data_df, sensor_columns):
    

    df_calibrated = ham_data_df.copy()

    pay = df_calibrated[sensor_columns] - BLACK_REF
    payda = WHITE_REF - BLACK_REF

    payda[payda == 0] = 1 # Bölme işleminde sıfır hatasını önlemek için payda sıfır olanları 1 yapıyoruz

    df_calibrated[sensor_columns] = pay / payda

    df_calibrated[sensor_columns] = df_calibrated[sensor_columns].clip(0.0, 1.0)

    return df_calibrated

class KumasSpektralDataset(Dataset):

    def __init__(self, hibrit_kumas_veriset):
    
        # Veriyi oku
        self.data = pd.read_csv(hibrit_kumas_veriset)

        # # İlk 18 sütun isimlerini al
        sensor_columns = self.data.columns[:18]  

        #18 sütunu al
        calibrated_data = calibrate_data(self.data, sensor_columns)

        # Kalibre edilmiş verileri Pytorch okuyabilsin diye Numpy array formatına çeviriyoruz
        self.X = calibrated_data[sensor_columns].values

        # Label sütununu al
        self.Y = self.data.iloc[:, -1].astype(float).values

        print("Veri seti başarıyla yüklendi ve kalibre edildi.")

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
