# Constants
TARGET = 10
LEARNING_RATE = 0.1

 # Normalde random seçicez şimdilik kendim değer atadım
old_w_1 = 1
old_w_2 = 2
old_b = -1



def linear(x_1, x_2) : 

    z = old_w_1 * x_1 + old_w_2 * x_2 + old_b

    return z



def relu(z) :

    h = z if z > 0 else 0

    return h
    


def output(h) :

    return h



def forward(x_1, x_2) : 
    
    z = linear(x_1, x_2)
    h = relu(z)
    prediction = output(h)

    return z, h, prediction, x_1 , x_2, 



def loss_function(prediction) :

    total_loss = 1/2 * (TARGET - prediction) ** 2

    return total_loss



def chain_rule(x_1, x_2, z, prediction):

    total_loss_prime = prediction - TARGET
    
    h_prime = 1 if z > 0 else 0

    z_prime_with_respect_to_w_1 = x_1
    z_prime_with_respect_to_w_2 = x_2
    z_prime_with_respect_to_b = 1

    gradient_slope_w_1 = total_loss_prime * h_prime * z_prime_with_respect_to_w_1
    gradient_slope_w_2 = total_loss_prime * h_prime * z_prime_with_respect_to_w_2
    gradient_slope_b = total_loss_prime * h_prime * z_prime_with_respect_to_b


    return gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b



def backpropagation(x_1, x_2, z, prediction, old_w_1, old_w_2, old_b):

    total_loss = loss_function(prediction)
    gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b= chain_rule(x_1, x_2, z, prediction)
    step_size_w_1 = LEARNING_RATE * gradient_slope_w_1
    step_size_w_2 = LEARNING_RATE * gradient_slope_w_2
    step_size_b = LEARNING_RATE * gradient_slope_b


    new_w_1 = old_w_1 - step_size_w_1
    new_w_2 = old_w_2 - step_size_w_2
    new_b = old_b - step_size_b

    return total_loss, gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b, old_w_1, new_w_1, old_w_2, new_w_2, old_b, new_b



epochs = 5

for epoch in range(epochs) :
    z, h, prediction, x_1, x_2= forward(2,3)

    total_loss, gradient_slope_w_1, gradient_slope_w_2, gradient_slope_b, old_w_1, new_w_1, old_w_2, new_w_2, old_b, new_b= backpropagation(x_1, x_2, z, prediction, old_w_1, old_w_2, old_b)

    print(f"{epoch+1}. Tur hesaplanan prediction: {prediction} \n hesaplanan toplam hata: {total_loss} \n hesaplanan w_1 gradient slope: {gradient_slope_w_1} \n hesaplanan w_2 gradient slope: {gradient_slope_w_2} \n hesaplanan b gradient slope: {gradient_slope_b} \n hesaplanan   yeni w1 değeri: {new_w_1} \n hesaplanan   yeni w2 değeri: {new_w_2} \n hesaplanan   yeni b değeri: {new_b} ")

    old_w_1 = new_w_1
    old_w_2 = new_w_2
    old_b = new_b









