import random

# Constants
B_1 = -1
W_2 = 2
B_2 = 0.5
TARGET = 5
LEARNING_RATE = 0.1

old_w_1 = 1 # Normalde random seçicez şimdilik sabit bıraktım 




def linear(x) : 

    z_1 = old_w_1 * x + B_1

    return z_1



def relu(z_1) :

    h = z_1 if z_1 > 0 else 0

    return h
    


def output(h) :
    
    prediction = W_2 * h + B_2

    return prediction


def forward(x) : 
    
    z_1 = linear(x)
    h = relu(z_1)
    prediction = output(h)

    return z_1, h, prediction, x




def loss_function(prediction) :

    total_loss = 1/2 * (TARGET - prediction) ** 2

    return total_loss




def chain_rule(x, z_1, prediction):

    total_loss_prime = prediction - TARGET
    
    prediction_prime = W_2


    h_prime = 1 if z_1 > 0 else 0

    z_1_prime = x

    gradient_slope = total_loss_prime * prediction_prime * h_prime * z_1_prime


    return gradient_slope




def backpropagation(x, z_1, prediction, old_w_1):

    total_loss = loss_function(prediction)
    gradient_slope = chain_rule(x, z_1, prediction)
    step_size = LEARNING_RATE * gradient_slope


    new_w_1 = old_w_1 - step_size

    return total_loss, gradient_slope,old_w_1, new_w_1



epochs = 50

for epoch in range(epochs) :
    z_1, h, prediction, x = forward(2)

    total_loss, gradient_slope, old_w_1, new_w_1 = backpropagation(x,z_1,prediction,old_w_1)

    print(f"{epoch}. Tur hesaplanan prediction: {prediction} \n hesaplanan toplam hata: {total_loss} \n hesaplanan gradient slope: {gradient_slope}  yeni w1 değeri: {new_w_1} ")

    old_w_1 = new_w_1








