import numpy as np 
import random as rd
import function 

class Linear():
    def __init__(self, n_in, n_out):
        self.W = np.array([[rd.randint(-1000,1000)/1000 for i in range(n_in)] for j in range(n_out)])
        self.b = np.zeros((n_out,1))
    def forward(self, a_prev):
        self.a_prev = a_prev
        return np.dot(self.W,a_prev) + self.b
    def backward(self, delta):
        self.dW = 1/self.a_prev.shape[1] * np.dot(delta, np.transpose(self.a_prev))
        self.db = 1/self.a_prev.shape[1] * np.sum(delta, keepdims = True)
        return np.dot(np.transpose(self.W),delta)
     
class ReLU():
    def __init__(self):
        return
    def forward(self, z):
        self.z = z
        return function.Relu(z)
    def backward(self, delta):
        return delta*function.Relu_prime(self.z)
        
class Sigmoid():
    def __init__(self):
        return
    def forward(self, z):
        self.z = z
        return function.sigmoid(z)
    def backward(self, delta):
        return delta*function.sigmoid_prime(self.z)
    
class Network():
    def __init__(self, layers):
        self.layers = layers 
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
    def update(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr*layer.dW
                layer.b -= lr*layer.db

