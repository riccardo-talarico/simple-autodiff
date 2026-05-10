from autodiff.models import Model
from autodiff.layers import Linear
from autodiff.losses import MeanSquaredError
from autodiff.value import Value
from autodiff.datasets import make_linear_regression

linear_regression = Model([
    Linear((1,1),(1,1))
    ])

linear_regression.add_loss(MeanSquaredError())

X,y = make_linear_regression(low=0,high=1000,n_instances=100)

# SGD, batch_size = 1
linear_regression.train(X,y, 1, 20,Value(3e-2))

# They should be equal to [[2]] and [[1]]
for param in linear_regression.parameters:
    print(param.value)


for val in [Value(1),Value(2),Value(3),Value(4),Value(5)]:
    print(f"Input: {val.value}, 2x+1={linear_regression.predict(val)}")

linear_regression.plot_learning_curve()
