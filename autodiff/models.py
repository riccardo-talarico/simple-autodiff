from autodiff.layers import *
from autodiff.losses import *
from autodiff.datasets import *


def one_hot_encode(x:Value, num_classes: int):
    val = []
    for i in range(num_classes):
        if i==x.value:
            val.append(1)
        else:
            val.append(0)
    return Value(val).transpose()


class Model(object):

    def __init__(self, layers: List[Layer] = [], classification:bool=False):
        self.graph = ComputationalGraph()
        self.layers : List[Layer] = []
        self.loss = None
        self.output_node = None
        self.parameters : List[DataNode] = []
        self.classification = classification
        self.fitted = False
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
        if self.classification and self.n_classes:
            #debug: 
            #print(f"Not argmaxed output is {out.value}")
            out = Value(np.float128(np.argmax(out.value)))
            
        return out
    
    def _extract_num_classes(self, lis: List[Value]):
        num = 0
        lis = list(map(lambda x: x.value, lis))
        for i,el in enumerate(lis):
            if el not in lis[:i]:
                num += 1
        return num

    def train(self, X: List[Value], y: List[Value], batch_size : int, num_epochs: int, lr: Value):
        if self.classification:
            self.n_classes = self._extract_num_classes(y)
            y = list(map(lambda x: one_hot_encode(x, self.n_classes), y))
        i = 0
        total_instances = len(X)
        self.loss_history = []
        for epoch in range(num_epochs):
            epoch_loss = Value.ERROR
            for start in range(0, total_instances, batch_size):
                end = min(start + batch_size, total_instances)

                X_batch = X[start:end]
                y_batch = y[start:end]

                self.graph.zero_grad()
                self.loss.zero_grad()
                for x,y_ in zip(X_batch,y_batch):
                    self.input.value = x
                    self.loss.add_label_value(y_)
                    out = self.graph.forward()
                    out_loss = self.loss.forward()

                    loss_grad = self.loss.backward()
                    grad = self.graph.backward(grad_init=False)
                    #print(f"DEBUG, out_loss={out_loss}")
                    epoch_loss = epoch_loss + out_loss if epoch_loss is not Value.ERROR else out_loss
                
                self.step(lr * Value(1/len(X_batch)))
            epoch_loss = epoch_loss * Value(1/total_instances)
            self.loss_history.append(epoch_loss)
        self.fitted = True

    def plot_learning_curve(self):
        if not self.fitted:
            print("Cannot plot learning curve of untrained model.")
            return
        else:
            loss_history = list(map(lambda x: x.value, self.loss_history))
            loss_history = list(map(lambda x: x.reshape(-1), loss_history))
            plt.plot(loss_history)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Learning Curve")
            plt.show()