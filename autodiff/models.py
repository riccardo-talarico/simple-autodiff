from layers import *
from losses import *

class Model(object):

    def __init__(self, layers: List[Layer] = []):
        self.graph = ComputationalGraph()
        self.layers : List[Layer] = []
        self.loss = None
        self.output_node = None
        self.parameters : List[DataNode] = []
        self.input = DataNode(Value.ERROR, requires_grad=False)
        
        for layer in layers:
            added = self.add_layer(layer)
            if not added:
                print(f"WARNING: cannot add layer {layer}")
        
        
    def add_layer(self, layer : Layer) -> bool:
        
        if self.loss:
            print("WARNING: cannot add layers after adding the loss")
            return False
        if self.layers == []:
            parent = self.input
        else:
            parent = self.output_node
            #if not layer.check_input(parent):
            #    return False

        self.output_node = layer.build(parent, self.graph)

        self.layers.append(layer)
        self.parameters += layer.parameters
        return True
    
    def add_loss(self, loss : Loss) -> bool:
        
        if self.layers is []:
            parent = self.input
        else:
            parent = self.output_node
            self.graph.output_nodes.add(self.output_node)
        self.output_node = loss.build(parent)
        self.loss = loss
        return True


    def step(self, lr: Value):
        for param in self.parameters:
            param.value = param.value - lr * param.local_grad


    def predict(self, value : Value=None):
        self.input.value = value if value else self.input.value
        out = self.graph.forward()
        return out
    
    #TODO: fix this batch size issue
    def train(self, X: List[Value], y: List[Value], batch_size : int, num_epochs: int, lr: Value):
        i = 0
        for epoch in range(num_epochs):
            for x,y_ in zip(X[i:i+batch_size],y[i:i+batch_size]):
                self.graph.zero_grad()
                self.input.value = x
                self.loss.add_label_value(y_)
                out = self.graph.forward()
                out_loss = self.loss.forward()

                loss_grad = self.loss.backward()
                grad = self.graph.backward(grad_init=False)
                self.step(lr)
            i += (batch_size % len(X))

if __name__ == '__main__':
    linear_regression = Model([
        Linear((1,1),(1,1))
        ])
    
    linear_regression.add_loss(MeanSquaredError())

    # Linear Regression Dataset
    X = np.random.randint(low=0,high=1000,size=100)
    X = (X - np.mean(X,keepdims=True)) / np.std(X)
    y = 2*X+np.ones_like(X) 

    X,y = X.tolist(),y.tolist()

    X = list(map(lambda x: Value(x), X))
    y = list(map(lambda x: Value(x), y))

    #data = Dataset(X,y)

    
    for param in linear_regression.parameters:
        print(param.value)

    linear_regression.train(X,y,64,100,Value(1e-1))
    print(linear_regression.predict(Value(2)))
    for param in linear_regression.parameters:
        print(param.value)


    neural_net = Model([
        Linear((2,1),(5,1)),
        ReLu((5,1)),
        Linear((5,1),(1,1))
    ])

    neural_net.add_loss(MeanSquaredError())
    X = [np.array([[1,0]]).T, np.array([[0,0]]).T, np.array([[0,1]]).T,np.array([[1,1]]).T]
    y = [np.array([[1]]),np.array([[0]]),np.array([[1]]),np.array([[0]])]

    X = list(map(lambda x: Value(x), X))
    y = list(map(lambda x: Value(x), y))


    print(f"Params before:")
    for param in neural_net.parameters:
        print(param.value)

    neural_net.train(X,y, 4, 1500, Value(5e-2))
    print("Params after:")
    for param in neural_net.parameters:
        print(param.value)

    for x in X:
        print(f"Value: {x}\n prediction: {neural_net.predict(x)}")
    

               
