import random
import math

LEARNING_RATE = 0.001

# --- KAIMING (HE) INITIALIZATION ---

# 1. Katman (Gizli Nöronlar) için Standart Sapma: sqrt(2 / girdi_sayısı)
std_hidden = math.sqrt(2.0 / 2) # n_in = 2 (x1 ve x2)

old_w_1 = random.gauss(0, std_hidden) # Çan eğrisi etrafında kontrollü rastgele
old_w_2 = random.gauss(0, std_hidden)
old_b_1 = 0.0  # Bias'lar KAOSA YOL AÇMAMAK İÇİN HER ZAMAN SIFIRLA BAŞLAR!

old_w_3 = random.gauss(0, std_hidden)
old_w_4 = random.gauss(0, std_hidden)
old_b_2 = 0.0

# 2. Katman (Çıktı Nöronu) için Standart Sapma
std_output = math.sqrt(2.0 / 2) # n_in = 2 (h1 ve h2)

old_w_5 = random.gauss(0, std_output)
old_w_6 = random.gauss(0, std_output)
old_b_3 = 0.0


# Ai dan aldığım basit bir dataset
dataset = [
    (6.0, 8.0, 9.5),  # Öğrenci 1 (İdeal): Güzel çalışmış, uykusunu tam almış. Yüksek puan.
    (1.0, 9.0, 3.0),  # Öğrenci 2 (Tembel): Sadece uyumuş, hiç çalışmamış. Düşük puan.
    (4.0, 6.0, 6.5),  # Öğrenci 3 (Ortalama): Ortalama bir çalışma ve uyku. Ortalama puan.
    (10.0, 1.0, 4.0)  # Öğrenci 4 (ZOMBİ - TUZAK): İnanılmaz çok çalışmış ama uykusuzluktan sınavda bayılmış!
]


# Lineer katman 2 nöron 2 input

def neuron_1(x_1, x_2) : 

    z_1 = old_w_1 * x_1 + old_w_2 * x_2 + old_b_1

    return z_1



def neuron_2(x_1, x_2) : 

    z_2 = old_w_3 * x_1 + old_w_4 * x_2 + old_b_2

    return z_2



# Aktivasyon fonskiyonu
def relu(z) :

    # reLU eğer değer negatifse sıfır yapar pozitifse aynı bırakır
    h = z if z > 0 else 0

    return h
    

# Outputumuz tek nöronlu bir ağ olduğu için burda işlem yok
def output(h_1, h_2) :

    prediction = old_w_5 * h_1 + old_w_6 * h_2 + old_b_3

    return prediction


# Modelin öğreneceği datanın hangi yönde akacağını tanımlıyoruz
def forward(x_1, x_2) : 
    
    z_1 = neuron_1(x_1, x_2)
    z_2 = neuron_2(x_1, x_2)
    h_1 = relu(z_1)
    h_2 = relu(z_2)
    

    prediction = output(h_1, h_2)

    return z_1, h_1, z_2, h_2, prediction, x_1 , x_2, 


# Kayıp hesaplama metodumuz türevde kolaylık açısından başına 1/2 attık
def loss_function(target,prediction) :

    total_loss = 1/2 * (target - prediction) ** 2

    return total_loss


# Zincir kuralı ile türev hesapları her parametre için ayrı ayrı kısmi türev ve gradient slopeların hesaplanması
def chain_rule(x_1, x_2, z_1, z_2,h_1, h_2,prediction, target):

    total_loss_prime = prediction - target
    
    h_1_prime = 1 if z_1 > 0 else 0
    h_2_prime = 1 if z_2 > 0 else 0


    z_1_prime_with_respect_to_w_1 = x_1
    z_1_prime_with_respect_to_w_2 = x_2
    z_1_prime_with_respect_to_b_1 = 1

    z_2_prime_with_respect_to_w_3 = x_1
    z_2_prime_with_respect_to_w_4 = x_2
    z_2_prime_with_respect_to_b_2 = 1

    z_prime_with_respect_to_w_5 = h_1
    z_prime_with_respect_to_w_6 = h_2
    z_prime_with_respect_to_b_3 = 1
    


    # 1. Nörona giden hata, w_5 köprüsünden geçerek geri dönmeli!
    gradient_slope_w_1 = total_loss_prime * old_w_5 * h_1_prime * z_1_prime_with_respect_to_w_1
    gradient_slope_w_2 = total_loss_prime * old_w_5 * h_1_prime * z_1_prime_with_respect_to_w_2
    gradient_slope_b_1 = total_loss_prime * old_w_5 * h_1_prime * z_1_prime_with_respect_to_b_1

    # 2. Nörona giden hata, w_6 köprüsünden geçerek geri dönmeli!
    gradient_slope_w_3 = total_loss_prime * old_w_6 * h_2_prime * z_2_prime_with_respect_to_w_3
    gradient_slope_w_4 = total_loss_prime * old_w_6 * h_2_prime * z_2_prime_with_respect_to_w_4
    gradient_slope_b_2 = total_loss_prime * old_w_6 * h_2_prime * z_2_prime_with_respect_to_b_2

    gradient_slope_w_5 = total_loss_prime * z_prime_with_respect_to_w_5
    gradient_slope_w_6 = total_loss_prime * z_prime_with_respect_to_w_6
    gradient_slope_b_3 = total_loss_prime * z_prime_with_respect_to_b_3



    return gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b_1, gradient_slope_w_3, gradient_slope_w_4, gradient_slope_b_2, gradient_slope_w_5, gradient_slope_w_6, gradient_slope_b_3



# Her parametre için step sizeların hesaplanıp güncel parametlerin hesaplanması
def backpropagation(x_1, x_2, h_1, h_2, z_1, z_2, target, prediction, old_w_1, old_w_2, old_w_3, old_w_4, old_w_5, old_w_6, old_b_1, old_b_2, old_b_3):

    total_loss = loss_function(target,prediction)
    gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b_1, gradient_slope_w_3, gradient_slope_w_4, gradient_slope_b_2, gradient_slope_w_5, gradient_slope_w_6, gradient_slope_b_3 = chain_rule(x_1, x_2, z_1, z_2,h_1, h_2,prediction, target)
    
    step_size_w_1 = LEARNING_RATE * gradient_slope_w_1
    step_size_w_2 = LEARNING_RATE * gradient_slope_w_2
    step_size_b_1 = LEARNING_RATE * gradient_slope_b_1

    step_size_w_3 = LEARNING_RATE * gradient_slope_w_3
    step_size_w_4 = LEARNING_RATE * gradient_slope_w_4
    step_size_b_2 = LEARNING_RATE * gradient_slope_b_2

    step_size_w_5 = LEARNING_RATE * gradient_slope_w_5
    step_size_w_6 = LEARNING_RATE * gradient_slope_w_6
    step_size_b_3 = LEARNING_RATE * gradient_slope_b_3



    new_w_1 = old_w_1 - step_size_w_1
    new_w_2 = old_w_2 - step_size_w_2
    new_b_1 = old_b_1 - step_size_b_1

    new_w_3 = old_w_3 - step_size_w_3
    new_w_4 = old_w_4 - step_size_w_4
    new_b_2 = old_b_2 - step_size_b_2

    new_w_5 = old_w_5 - step_size_w_5
    new_w_6 = old_w_6 - step_size_w_6
    new_b_3 = old_b_3 - step_size_b_3


    return total_loss, gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b_1, gradient_slope_w_3, gradient_slope_w_4, gradient_slope_b_2, gradient_slope_w_5, gradient_slope_w_6, gradient_slope_b_3, old_w_1, new_w_1, old_w_2, new_w_2, old_b_1, new_b_1, old_w_3, new_w_3, old_w_4, new_w_4, old_b_2, new_b_2, old_w_5, new_w_5, old_w_6, new_w_6, old_b_3, new_b_3


# Modeli 100 tur eğitiyoruz
epochs = 13

for epoch in range(epochs) :
    # Her 1 turda her bütün dataseti sırayla gösteriyoruz
    for i in range(len(dataset)):
        
        z_1, h_1, z_2, h_2, prediction, x_1 , x_2, = forward(dataset[i][0], dataset[i][1])

        total_loss, gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b_1, gradient_slope_w_3, gradient_slope_w_4, gradient_slope_b_2, gradient_slope_w_5, gradient_slope_w_6, gradient_slope_b_3, old_w_1, new_w_1, old_w_2, new_w_2, old_b_1, new_b_1, old_w_3, new_w_3, old_w_4, new_w_4, old_b_2, new_b_2, old_w_5, new_w_5, old_w_6, new_w_6, old_b_3, new_b_3 = backpropagation(x_1, x_2, z_1, z_2, h_1, h_2, dataset[i][2], prediction, old_w_1, old_w_2, old_w_3, old_w_4, old_w_5, old_w_6, old_b_1, old_b_2, old_b_3)

        print(f"{epoch+1}. Tur {dataset[i]}. data için hesaplanan prediction: {prediction} \n hesaplanan toplam hata: {total_loss} \n hesaplanan w_1 gradient slope: {gradient_slope_w_1} \n hesaplanan w_2 gradient slope: {gradient_slope_w_2} \n hesaplanan b_1 gradient slope: {gradient_slope_b_1} \n hesaplanan   yeni w1 değeri: {new_w_1} \n hesaplanan   yeni w2 değeri: {new_w_2} \n hesaplanan   yeni b_1 değeri: {new_b_1}  hesaplanan w_3 gradient slope: {gradient_slope_w_3} \n hesaplanan w_4 gradient slope: {gradient_slope_w_4} \n hesaplanan b_2 gradient slope: {gradient_slope_b_2} \n hesaplanan yeni w_3 değeri: {new_w_3} \n hesaplanan yeni w_4 değeri: {new_w_4} \n hesaplanan yeni b_2 değeri: {new_b_2} hesaplanan w_5 gradient slope: {gradient_slope_w_5} \n hesaplanan w_6 gradient slope: {gradient_slope_w_6} \n hesaplanan b_3 gradient slope: {gradient_slope_b_3} \n hesaplanan yeni w_5 değeri: {new_w_5} \n hesaplanan yeni w_6 değeri: {new_w_6} \n hesaplanan yeni b_3 değeri: {new_b_3}")

        old_w_1 = new_w_1
        old_w_2 = new_w_2
        old_b_1 = new_b_1

        old_w_3 = new_w_3
        old_w_4 = new_w_4
        old_b_2 = new_b_2

        old_w_5 = new_w_5
        old_w_6 = new_w_6
        old_b_3 = new_b_3



print("--------------------------------------------------")
print("Tek bir data ile eğitilmiş modeli test ediyoruz")

# Modelin daha önce görmediği iki input
new_x_1 = 20
new_x_2 = 0

# Test aşamasında sadece forward metodunu çağırıyoruz
test_z_1, test_h_1, test_z_2, test_h_2, test_prediction, _, _ = forward(new_x_1, new_x_2)

print(f"Gelen Yeni Veriler -> x_1: {new_x_1}, x_2: {new_x_2}")
print(f"Eğitilmiş Modelin Tahmini: {test_prediction}")



