import numpy as np 
import function as func
import losses as ls
import network as nw
import matplotlib.pyplot as plt


def train(network, loss_fn, X, y, lr, epochs):
    for _ in range(epochs):
        y_hat = network.forward(np.transpose(X))
        loss = loss_fn.forward(y_hat,y)
        if _%100 == 0:
            print(loss)
        delta = loss_fn.backward(y_hat,y)
        network.backward(delta)
        network.update(lr)


N = 100
X, y = func.make_spirals(N)

# NN
model = nw.Network([
    nw.Linear(2, 64),
    nw.ReLU(),
    nw.Linear(64, 128),
    nw.ReLU(),
    nw.Linear(128, 1),
    nw.Sigmoid()
])


loss_fn = ls.BCE()
lr = 0.01
epochs = 50000

train(model, loss_fn, X, y, lr, epochs)

# Grille
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid = np.c_[xx.ravel(), yy.ravel()]
preds = model.forward(grid.T).reshape(xx.shape)

# Plot
plt.contour(xx, yy, preds, levels=[0.5], colors='black')
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdBu', edgecolors='k', s=20)
plt.title("Frontière de décision")
plt.show()