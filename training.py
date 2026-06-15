import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataPipeline import KumasSpektralDataset
from Fnn import SpektralModel

dataset = KumasSpektralDataset("hibrit_kumas_veriseti.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) #shuffle=True veriyi karıştırır her epochta farklı sırayla verir ezberlemeyi önler 
model = SpektralModel()

# Kayıp fonksiyonu ve optimizasyon algoritması
criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss, ikili sınıflandırma için uygun bir kayıp fonksiyonu

optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizasyon algoritması lr öğrenme oranı tamirci adımlarının büyüklüğü

# Eğitim döngüsü
num_epochs = 50 # Epoch sayısı, tüm veri setinin model tarafından kaç kez işleneceğini belirler

for epoch in range(num_epochs):
    total_loss = 0

    # DataLoader ile veriyi batchler halinde al
    for batch_x, batch_y in dataloader:
        # Modeli çalıştır ve tahminleri al
        # Modelin çıktısı [32, 1] boyutunda olacak çünkü batch_size 32 ve modelin çıktısı 1 bunu [32] boyutuna indirgemek için squeeze kullanacağız

        tahminler = model(batch_x).squeeze(1)

        # Kayıp fonksiyonunu hesapla
        loss = criterion(tahminler, batch_y)

        # Geri yayılım ve optimizasyon adımları

        optimizer.zero_grad() # Önceki adımın gradyanlarını sıfırla
        loss.backward() # Geri yayılım ile gradyanları hesapla
        optimizer.step() # Ağırlıkları güncelle

        total_loss += loss.item() # Kayıp değerini topla

    # Her epoch sonunda ortalama kaybı yazdır
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}") 