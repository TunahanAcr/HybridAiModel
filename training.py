import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataPipeline import KumasSpektralDataset
from Fnn import SpektralModel

dataset = KumasSpektralDataset("hibrit_kumas_veriseti.csv") # Datasetimizi fonksiyonumuzu yükledik
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # Dataloader fonksiyonumuzu yükledik, shuffle=True veriyi karıştırır her epochta farklı sırayla verir ezberlemeyi önler 
model = SpektralModel() # Modelimizi yükledik 

# Loss fonksiyonu ve optimizasyon algoritması
criterion = nn.CrossEntropyLoss() # Cross Entropy Loss, ikili sınıflandırma için uygun bir kayıp fonksiyonu içinde hem softmax hem de cross entropi fonksiyonları var

optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizasyon algoritması model.parameters optimizere, güncellenmesi gereken ağırlık ve bias değişkenlerinin referanslarını verir. Optimizer, sadece bu listedeki değişkenleri günceller. bir de learning rate i verdik

# Eğitim döngüsü
num_epochs = 50 # Epoch sayısı, tüm veri setinin model tarafından kaç kez işleneceğini belirler

for epoch in range(num_epochs):
    total_loss = 0

    # DataLoader ile veriyi batchler halinde al
    for batch_x, batch_y in dataloader:
        # Modeli çalıştır ve tahminleri al

        tahminler = model(batch_x)# Dataloaderın datesetten aldığı 32 batchlik tensoru modele verip değişkene koyduk

        # Kayıp fonksiyonunu hesapla
        loss = criterion(tahminler, batch_y) # Burda batch_y ile targetı tahminler ile modelin tahmin ettiği değerleri loss fonksiyonumuza hesaplattırıp değişkene atıyoruz 

        # Backpropagation ve optimizasyon adımları

        optimizer.zero_grad() # Önceki adımın gradyanlarını sıfırla pytorchta klasik
        loss.backward() # Backpropagation ile gradyanları hesaplat
        optimizer.step() # Gradian slope ve lr değerlerini çarpıp stepi hesaplayıp weights ve biasleri günceller  her adımda bütün 641 parametreyi günceller

        total_loss += loss.item() # Bu işlem, o turdaki ortalama hatayı hesaplayıp ekrana yazdırmak içindir. Modelin öğrenmesine bir etkisi yoktur, sadece eğitim sürecinin ilerleyişini takip için

    # Her epoch sonunda ortalama kaybı yazdır
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}") 