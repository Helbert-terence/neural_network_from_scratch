import numpy as np 

def make_spirals(n, rev = 4, ampl = 5):
    """
    n : int

    return : X : numpy array (2n,2) of coordinates x,y that form two spirals 
             Labels : numpy array (2n,1) of label that specifie if a point of X is from spiral 0 or 1
    """
    X = []
    Labels = []
    for i in range(1, n):
        x1 = ampl*i/n * np.cos(rev*np.pi*i/n)
        y1 = ampl*i/n * np.sin(rev*np.pi*i/n)
        x2 = ampl*i/n * np.cos(rev*np.pi*i/n + np.pi)
        y2 = ampl*i/n * np.sin(rev*np.pi*i/n + np.pi)
        X.append([x1,y1])
        X.append([x2,y2])
        Labels.append([0])
        Labels.append([1])
    return np.array(X),np.array(Labels).reshape(-1, 1)
    
def sigmoid(z):
    """
    z : numpy array 
    """
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    """
    z : numpy array 
    """
    return sigmoid(z)*(1-sigmoid(z))

def Relu(z):
    """
    z : numpy array 
    """
    return np.maximum(0,z)

def Relu_prime(z):
    """
    z : numpy array 
    """
    return np.where(z<0,0,1)