from autodiff.computational_graph import *
from autodiff.node import DataNode, SubNode, CustomNode, MultNode
import numpy as np


class Loss():
    def __init__(self):
        # Placeholder value of -1 for the label
        self.label_node : DataNode = DataNode(Value(-1), requires_grad=False)
        self.graph = ComputationalGraph()
    def build(self):
        pass
    def add_label_value(self, value : Value):
        self.label_node.value = value

    def forward(self):
        return self.graph.forward()

    def backward(self):
        #self.graph.zero_grad()
        return self.graph.backward()
    
    def zero_grad(self):
        self.graph.zero_grad()




def forward_mean(self, input:List[Value]):
    self.x = input[0]
    self.N = np.prod(self.x.shape)
    return Value(np.sum(self.x.value) / self.N)

def backward_mean(self, inputs:List[Value], upstream_grad:Value):
    return [upstream_grad * Value(np.ones_like(self.x.value) * (1/self.N))]


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()
        self.sub = SubNode()
        self.mean = CustomNode(
            requires_grad=True,
            forward_op=forward_mean,
            backward_op= backward_mean
        )
        self.squared = MultNode()
        self.output_node = self.mean
    
    def build(self, input_node:Node):
        self.graph.add_edge(input_node, self.sub)
        self.graph.add_edge(self.label_node, self.sub)
        self.graph.add_edge(self.sub,self.squared)
        self.graph.add_edge(self.sub,self.squared)
        self.graph.add_edge(self.squared, self.mean)
        self.graph.add_nodes([self.label_node,self.mean,self.squared,self.sub])
        self.graph.output_nodes.add(self.mean)
        
        return self.output_node


def forward_ce(self: Node, input:List[Value]):
    self.y = input[0].value
    self.p = input[1].value
    eps = 1e-12
    self.cross_entropy = -np.sum(self.y * np.log(self.p+eps))
    return Value(self.cross_entropy)

def backward_ce(self:Node, inputs:List[Value], upstream_grad:Value):
    # softmax and sigmoid optimization
    return [Value(self.p - self.y), Value(self.p - self.y)]

# L = - y log y + (1-y)log(1-y)
class CrossEntropy(Loss):
    """
    Note: requires sigmoid or softmax activation functions
    """
    def __init__(self):
        super().__init__()
        self.cross_entropy_node = CustomNode(
            requires_grad=True,
            forward_op=forward_ce,
            backward_op= backward_ce
        )
        self.output_node = self.cross_entropy_node
    
    def build(self, input_node:Node):
        self.graph.add_edge(self.label_node, self.cross_entropy_node)
        self.graph.add_edge(input_node, self.cross_entropy_node)
        self.graph.add_node(self.cross_entropy_node, is_output=True, is_input=True)
        return self.output_node
    

