import numpy as np 

class Loss:
    # class for inheritance for all loss functions
    def __init__(self):
        return
    def forward(self, y_hat, y):
        raise NotImplementedError
    def backward(self, y_hat, y):
        raise NotImplementedError
    
class BCE(Loss):
    # Binary cross entropy : standart loss for binary classification
    def __init__(self):
        return
    def forward(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
        L = 0
        m = len(y)
        y_hat = np.transpose(y_hat)
        for i in range(len(y)):
            L -= 1/m*(y[i]*np.log(y_hat[i])+(1-y[i])*np.log(1-y_hat[i]))
        return L
    def backward(self, y_hat, y):
        # derivative of BCE with respect to y_hat
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
        y = np.transpose(y)
        return (y_hat - y)/(y_hat*(1 - y_hat))
