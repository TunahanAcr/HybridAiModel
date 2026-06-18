# Constants
LEARNING_RATE = 0.001

 # Normalde random seçicez şimdilik kendim değer atadım
old_w_1 = 1
old_w_2 = 2
old_b = -1

# Ai dan aldığım basit bir dataset
dataset = [
    (6.0, 8.0, 9.5),  # Öğrenci 1 (İdeal): Güzel çalışmış, uykusunu tam almış. Yüksek puan.
    (1.0, 9.0, 3.0),  # Öğrenci 2 (Tembel): Sadece uyumuş, hiç çalışmamış. Düşük puan.
    (4.0, 6.0, 6.5),  # Öğrenci 3 (Ortalama): Ortalama bir çalışma ve uyku. Ortalama puan.
    (10.0, 1.0, 4.0)  # Öğrenci 4 (ZOMBİ - TUZAK): İnanılmaz çok çalışmış ama uykusuzluktan sınavda bayılmış!
]


# Lineer katman
def linear(x_1, x_2) : 

    z = old_w_1 * x_1 + old_w_2 * x_2 + old_b

    return z


# Aktivasyon fonskiyonu
def relu(z) :

    # reLU eğer değer negatifse sıfır yapar pozitifse aynı bırakır
    h = z if z > 0 else 0

    return h
    

# Outputumuz tek nöronlu bir ağ olduğu için burda işlem yok
def output(h) :

    prediction = h

    return prediction


# Modelin öğreneceği datanın hangi yönde akacağını tanımlıyoruz
def forward(x_1, x_2) : 
    
    z = linear(x_1, x_2)
    h = relu(z)
    prediction = output(h)

    return z, h, prediction, x_1 , x_2, 


# Kayıp hesaplama metodumuz türevde kolaylık açısından başına 1/2 attık
def loss_function(target,prediction) :

    total_loss = 1/2 * (target - prediction) ** 2

    return total_loss


# Zincir kuralı ile türev hesapları her parametre için ayrı ayrı kısmi türev ve gradient slopeların hesaplanması
def chain_rule(x_1, x_2, z, prediction, target):

    total_loss_prime = prediction - target
    
    h_prime = 1 if z > 0 else 0

    z_prime_with_respect_to_w_1 = x_1
    z_prime_with_respect_to_w_2 = x_2
    z_prime_with_respect_to_b = 1

    gradient_slope_w_1 = total_loss_prime * h_prime * z_prime_with_respect_to_w_1
    gradient_slope_w_2 = total_loss_prime * h_prime * z_prime_with_respect_to_w_2
    gradient_slope_b = total_loss_prime * h_prime * z_prime_with_respect_to_b


    return gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b


# Her parametre için step sizeların hesaplanıp güncel parametlerin hesaplanması
def backpropagation(x_1, x_2, z,target, prediction, old_w_1, old_w_2, old_b):

    total_loss = loss_function(target,prediction)
    gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b= chain_rule(x_1, x_2, z, prediction, target)
    step_size_w_1 = LEARNING_RATE * gradient_slope_w_1
    step_size_w_2 = LEARNING_RATE * gradient_slope_w_2
    step_size_b = LEARNING_RATE * gradient_slope_b


    new_w_1 = old_w_1 - step_size_w_1
    new_w_2 = old_w_2 - step_size_w_2
    new_b = old_b - step_size_b

    return total_loss, gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b, old_w_1, new_w_1, old_w_2, new_w_2, old_b, new_b


# Modeli 100 tur eğitiyoruz
epochs = 100

for epoch in range(epochs) :
    # Her 1 turda her bütün dataseti sırayla gösteriyoruz
    for i in range(len(dataset)):
        
        z, h, prediction, x_1, x_2= forward(dataset[i][0], dataset[i][1])

        total_loss, gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b, old_w_1, new_w_1, old_w_2, new_w_2, old_b, new_b= backpropagation(x_1, x_2, z,dataset[i][2], prediction, old_w_1, old_w_2, old_b)

        print(f"{epoch+1}. Tur {dataset[i]}. data için hesaplanan prediction: {prediction} \n hesaplanan toplam hata: {total_loss} \n hesaplanan w_1 gradient slope: {gradient_slope_w_1} \n hesaplanan w_2 gradient slope: {gradient_slope_w_2} \n hesaplanan b gradient slope: {gradient_slope_b} \n hesaplanan   yeni w1 değeri: {new_w_1} \n hesaplanan   yeni w2 değeri: {new_w_2} \n hesaplanan   yeni b değeri: {new_b} ")

        old_w_1 = new_w_1
        old_w_2 = new_w_2
        old_b = new_b


print("--------------------------------------------------")
print("Tek bir data ile eğitilmiş modeli test ediyoruz")

# Modelin daha önce görmediği iki input
new_x_1 = 20
new_x_2 = 0

# Test aşamasında sadece forward metodunu çağırıyoruz
test_z, test_h, test_prediction, _, _ = forward(new_x_1, new_x_2)

print(f"Gelen Yeni Veriler -> x_1: {new_x_1}, x_2: {new_x_2}")
print(f"Eğitilmiş Modelin Tahmini: {test_prediction}")









