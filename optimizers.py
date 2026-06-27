import numpy as np

class Adam:
    # adam optimizer to have better convergence
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 # iteration
    def update(self, network):
        self.t += 1
        for layer in network.layers:
            if not hasattr(layer, 'mW'):
                layer.mW = np.zeros_like(layer.W)
                layer.vW = np.zeros_like(layer.W)
                layer.mb = np.zeros_like(layer.b)
                layer.vb = np.zeros_like(layer.b)
            layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
            layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * layer.dW**2
            layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db
            layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * layer.db**2

            mW_hat = layer.mW / (1 - self.beta1**self.t)
            vW_hat = layer.vW / (1 - self.beta2**self.t)
            mb_hat = layer.mb / (1 - self.beta1**self.t)
            vb_hat = layer.vb / (1 - self.beta2**self.t)

            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)