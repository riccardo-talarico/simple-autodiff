from autodiff.models import *
from autodiff.layers import Linear, SoftMax
from autodiff.losses import CrossEntropy

classifier = Model(
    [
        Linear((2,1),(3,1), init_mode='he'),
        SoftMax((3,1))
    ], classification=True
)

classifier.add_loss(CrossEntropy())

X,y = make_blobs_3(n_samples=200,show_blobs=True)
X_train,y_train = X[:150],y[:150]
X_test,y_test = X[150:],y[150:]

print(f"Training set has {len(X_train)} instances, test set has {len(X_test)}")

classifier.train(X_train,y_train, 1, 80, Value(1e-2))

acc=0
for x,y_true in zip(X_train,y_train):
    y_pred = classifier.predict(x)
    if y_pred == y_true: 
        acc += 1
print(f"Train accuracy: {acc*(2/3)}%")


acc = 0
for x,y_true in zip(X_test,y_test):
    y_pred = classifier.predict(x)
    if y_pred == y_true:
        acc += 1
print(f"Test accuracy: {acc*(100/len(X_test))}%")

classifier.plot_learning_curve()