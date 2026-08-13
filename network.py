import numpy as np 
import random as rd
import function 

class Linear():
    # fully connected layer 
    def __init__(self, n_in, n_out):
        self.W = np.array([[rd.randint(-1000,1000)/1000 for i in range(n_in)] for j in range(n_out)]) # weight, initially randomize between -1 and 1, shape (n_out, n_in), 1 row for each output neurons, 1 column for each entry
        self.b = np.zeros((n_out,1)) # bias
    def forward(self, a_prev):
        self.a_prev = a_prev # keep for backprop
        return np.dot(self.W,a_prev) + self.b
    def backward(self, delta):
        self.dW = np.dot(delta, np.transpose(self.a_prev))
        self.db = np.sum(delta, axis = 1, keepdims = True)
        return np.dot(np.transpose(self.W),delta) # to give to previous 
     
class ReLU():
    # layer ReLU
    def __init__(self):
        return
    def forward(self, z):
        self.z = z
        return function.Relu(z)
    def backward(self, delta):
        return delta*function.Relu_prime(self.z)
        
class Sigmoid():
    # layer sigmoid
    def __init__(self):
        return
    def forward(self, z):
        self.z = z
        return function.sigmoid(z)
    def backward(self, delta):
        return delta*function.sigmoid_prime(self.z)
    
class Network():
    # to arrange multiple layers 
    def __init__(self, layers):
        self.layers = layers 
    def forward(self, X):
        # forward pass 
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, delta):
        # back prop 
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
    def update(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr*layer.dW
                layer.b -= lr*layer.db

