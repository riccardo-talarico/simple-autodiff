from computational_graph import *
from node import DataNode, SubNode, CustomNode, MultNode
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
        #self.graph.add_node(input_node)
        self.graph.add_edge(input_node, self.sub)
        self.graph.add_edge(self.label_node, self.sub)
        self.graph.add_edge(self.sub,self.squared)
        self.graph.add_edge(self.sub,self.squared)
        self.graph.add_edge(self.squared, self.mean)
        self.graph.add_nodes([self.label_node,self.mean,self.squared,self.sub])
        self.graph.output_nodes.add(self.mean)
        
        return self.output_node
    
    def forward(self):
        return self.graph.forward()

    def backward(self):
        self.graph.zero_grad()
        return self.graph.backward()
