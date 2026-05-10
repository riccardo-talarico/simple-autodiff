from autodiff.computational_graph import *
from typing import Tuple, Literal
from autodiff.node import DataNode, MatrixMultNode, AddNode, CustomNode
import numpy as np


class Layer(object):
    def __init__(self):
        self.nodes = []
        self.parameters : List[DataNode] = []

    def build(self, input_node: Node, graph: ComputationalGraph) -> Node:
        """
        Attaches the input nodes to the layer.
        It then returns the output node of the layer
        
        Args:
            input_node ([Node]): input node to use as parent to the layer.
            graph (ComputationalGraph): the computational graph on which you want to add the layer.

        Returns:
            (Node) : output node of the layer. 
        """
        return Node(False)
    
    def check_input(self, input : Node) -> bool:
        """
        Checks if the input node is compatible with the layer expected input shape.

        Args:
            input (Node): input node that will be the parent of the layer.
        Returns:
            (bool): True if the input is compatible
        """
        return True


class Linear(Layer):
    def __init__(
            self, 
            in_shape, 
            out_shape, 
            init_values: Tuple[Value,Value] | None = None, 
            init_mode : Literal['random','he'] ='random'
            ):
        
        self.out_shape = out_shape
        self.in_shape = in_shape
        if init_values:
            self.check_init_values_shape(init_values)
        if init_values:
            W,b = init_values
        else:
            if init_mode == 'random':
                W, b = self.rand_init(in_shape, out_shape)
            elif init_mode == 'he':
                W,b = self.he_init(in_shape, out_shape) 
            else:
                print("Warning: unrecognized init_mode: {init_mode}. Defaulting to random.")
                W,b = self.rand_init(in_shape, out_shape)
        self.weight_matrix = DataNode(W, requires_grad = True)
        self.bias = DataNode(b, requires_grad = True)
        self.parameters = [self.weight_matrix, self.bias]

        self.mult_node = MatrixMultNode()
        self.addition_node = AddNode()
        self.output_node = self.addition_node

    def check_input(self, input : Node):
        return input.shape == self.in_shape
    
    def he_init(self, in_shape, out_shape) -> Tuple[Value,Value]:
        self.matrix_shape = (out_shape[0], in_shape[0])
        self.bias_shape = out_shape
        fan_in = self.matrix_shape[0]
        std = (2.0 / fan_in) ** 0.5
        W = np.random.randn(*self.matrix_shape) * std
        b = np.zeros(self.bias_shape)
        return Value(W),Value(b)

    def rand_init(self, in_shape, out_shape) -> Tuple[Value,Value]:
        self.matrix_shape = (out_shape[0], in_shape[0])
        self.bias_shape = out_shape
        W = np.random.rand(self.matrix_shape[0],self.matrix_shape[1])
        b = np.random.rand(self.bias_shape[0],self.bias_shape[1])
        return Value(W),Value(b)
    
    def build(self, input_node : Node, graph: ComputationalGraph) -> Node:
       
        # c = W x
        graph.add_edge(self.weight_matrix, self.mult_node)
        graph.add_edge(input_node, self.mult_node)
        # out = b + c
        graph.add_edge(self.mult_node, self.addition_node)
        graph.add_edge(self.bias, self.addition_node)

        graph.add_node(self.weight_matrix)
        graph.add_node(self.bias)
        graph.add_node(self.mult_node)
        graph.add_node(self.addition_node)

        return self.output_node


class ActivationLayer(Layer):
    def __init__(self, shape):
        self.in_shape = shape
        self.output_node = Node(True)
        self.output_node.shape = shape
        self.parameters = []
    
    def build(self, input_node: Node, graph: ComputationalGraph):
        graph.add_edge(input_node, self.output_node)
        graph.add_node(self.output_node)
        return self.output_node



def forward_relu(self: Node, input: List[Value]):
    input = input[0]
    self.mask = Value(input.value > 0)
    return Value(np.maximum(0,input.value))

def backward_relu(self:Node, input:List[Value], upstream_grad:Value):
    return [upstream_grad * self.mask]

class ReLu(ActivationLayer):
    def __init__(self, shape):
        super().__init__(shape)
        self.relu = CustomNode(
            requires_grad=True,
            forward_op = forward_relu,
            backward_op = backward_relu
            )
        self.output_node = self.relu
    

def forward_softmax(self:Node, input: List[Value]):
    input = input[0]
    input_shifted = input.value - np.max(input.value)
    self.exp = np.exp(input_shifted)
    self.softmax = self.exp / np.sum(self.exp)
    return Value(self.softmax)
    
# vectorized formula is equivalent to Jacobian * upstream_gradient
def backward_softmax(self:Node, input: List[Value], upstream_grad:Value):
    upstream = upstream_grad.value
    return [Value(self.softmax * (upstream - np.sum(upstream * self.softmax)))]

# softmax(x_i) = e^x_i / sum_j (e^x_j)
class SoftMax(ActivationLayer):
    def __init__(self, shape):
        super().__init__(shape)
        self.softmax = CustomNode(
            requires_grad=True,
            forward_op = forward_softmax,
            backward_op = backward_softmax
            )
        self.output_node = self.softmax


if __name__ == '__main__':
    g = ComputationalGraph()
    v = Value([1,1,1])
    v.value = v.value.T
    input = DataNode(v, requires_grad=False)
    linear = Linear((3,1),(1,1), init_values=(Value([1,0,0]),Value(-2)))
    lin_out = linear.build(input, g)
    assert g.forward()==Value(-1)
    relu = ReLu((1,1))
    relu_out = relu.build(lin_out, g)
    assert g.forward() == Value(0)