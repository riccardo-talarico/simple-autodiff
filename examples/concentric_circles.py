from autodiff.models import *
from autodiff.losses import CrossEntropy
from autodiff.datasets import make_circles


classifier = Model([
    Linear((2,1),(32,1), init_mode='he'),
    ReLu((32,1)),
    Linear((32,1),(32,1), init_mode='he'),
    ReLu((32,1)),
    Linear((32,1),(2,1), init_mode='he'),
    SoftMax((2,1))
], classification=True)

classifier.add_loss(CrossEntropy())

X,y = make_circles(show_circles=True)

X_train,y_train = X[:150],y[:150]
X_test,y_test = X[150:200],y[150:200]


classifier.train(X_train,y_train, 1, 200, Value(1e-2))

acc = 0
for x,y_true in zip(X_train,y_train):
    y_pred = classifier.predict(x)
    if y_pred == y_true: 
        acc += 1
print(f"Train accuracy: {acc* (2/3)}%")

acc = 0
for x,y_true in zip(X_test,y_test):
    y_pred = classifier.predict(x)
    if y_pred == y_true: 
        acc += 1
print(f"Test Accuracy: {acc*2}%")

classifier.plot_learning_curve()

