from autodiff.models import Model
from autodiff.layers import Linear,ReLu
from autodiff.losses import MeanSquaredError
from autodiff.value import Value
import numpy as np

neural_net = Model([
    Linear((2,1),(8,1), init_mode='he'),
    ReLu((8,1)),
    Linear((8,1),(1,1), init_mode='random'),
    ReLu((1,1))
])

neural_net.add_loss(MeanSquaredError())

# Creating the XOR dataset
X = [np.array([[1,0]]).T, np.array([[0,0]]).T, np.array([[0,1]]).T,np.array([[1,1]]).T]
y = [np.array([[1]]),np.array([[0]]),np.array([[1]]),np.array([[0]])]

X = list(map(lambda x: Value(x), X))
y = list(map(lambda x: Value(x), y))


neural_net.train(X,y, 1, 300, Value(1e-2))

# Verify:
for x in X:
    print(f"Value: {x}\n prediction: {neural_net.predict(x)}")

neural_net.plot_learning_curve()