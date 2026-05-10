from autodiff.value import Value
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

# Blobs: higher separation means easier (linearly separable), lower noise means tighter clusters
def make_blobs_3(n_samples=150, separation=4.0, show_blobs:bool=False):
    n = n_samples // 3
    
    centers = [
        np.array([0, separation]),
        np.array([-separation, -separation]),
        np.array([separation, -separation])
    ]
    
    X_list = []
    y_list = []
    
    for i, c in enumerate(centers):
        Xi = c + np.random.randn(n, 2)
        yi = np.full(n, i)
        X_list.append(Xi)
        y_list.append(yi)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    if show_blobs:
        plt.scatter(X[:,0], X[:,1], c=y)
        plt.show()


    X = list(map(lambda x: Value(x).transpose(),X.tolist()))
    y = list(map(lambda x: Value(x),y.tolist()))

    # Random Shuffling
    zipping =list(zip(X,y))
    shuffle(zipping)
    X,y = zip(*zipping)
    
    return X, y


#Circles: higher noise means harder, lower factor means inner circle smaller, so an easier separation
def make_circles(n_samples=200, noise=0.05, factor=0.5, show_circles:bool=False):
    n = n_samples // 2
    
    # outer circle
    theta = 2 * np.pi * np.random.rand(n)
    outer = np.c_[np.cos(theta), np.sin(theta)]
    
    # inner circle
    theta = 2 * np.pi * np.random.rand(n)
    inner = factor * np.c_[np.cos(theta), np.sin(theta)]
    
    X = np.vstack([outer, inner])
    X += noise * np.random.randn(*X.shape)
    
    y = np.hstack([np.zeros(n), np.ones(n)])

    if show_circles:
        plt.scatter(X[:,0], X[:,1], c=y)
        plt.show()


    X = list(map(lambda x: Value(x).transpose(),X.tolist()))
    y = list(map(lambda x: Value(x),y.tolist()))
    
    # Random Shuffling
    zipping =list(zip(X,y))
    shuffle(zipping)
    X,y = zip(*zipping)
    
    return X, y


def make_linear_regression(low, high, n_instances):
    # Linear Regression Dataset
    X = np.random.randint(low=low,high=high,size=n_instances)
    X = (X - np.mean(X,keepdims=True)) / np.std(X)
    y = 2*X+np.ones_like(X) 

    X,y = X.tolist(),y.tolist()

    X = list(map(lambda x: Value(x), X))
    y = list(map(lambda x: Value(x), y))
    return X,y