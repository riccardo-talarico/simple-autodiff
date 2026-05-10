from autodiff.node import Node
from typing import List, Set
from graphlib import TopologicalSorter
from autodiff.value import Value


class ComputationalGraph(object):
    def __init__(self):
        self.nodes : List[Node] = []
        self.input_nodes : List[Node] = []
        self.output_nodes : Set[Node] = set()
        self.sorted : bool = False
        
    def add_node(self, node : Node, resort=True, is_input:bool=False, is_output:bool=False):
        """
        Adds 'node' to the graph.

        Args:
            node (Node): the node to add to the graph
            resort (bool): if true, applies topological sort to the graph after adding the nodes
            is_input (bool): if true, the node will be considered as input in the graph
            is_output (bool): if true, the node will be considered as output in the graph
        """
        self.nodes.append(node)
        if is_input:
            self.input_nodes.append(node)
        elif is_output:
            self.output_nodes.add(node)
        if resort:
            self.sort_topological()
        else:
            self.sorted=False

    def add_nodes(
            self, 
            nodes: List[Node], 
            resort = True, 
            input_idx: list | None = None, 
            output_idx : list | None = None
            ):
        """
        Adds a list of nodes to the graph.

        Args:
            nodes (List[Node]): the list of nodes to add.
            resort (bool): if true, the graph is topologically sorted after adding the ndoes.
            input_idx (list | None): list of indices of 'nodes' that must be considered as inputs.
            output_idx (list | None): list of indices of 'nodes that must be considered as outputs
        """
        for idx,node in enumerate(nodes):
            input = False
            output = False
            if input_idx and idx in input_idx:
                input = True
            if output_idx and idx in output_idx:
                output=True
            self.add_node(node,resort=False,is_input=input,is_output=output)
        if resort:
            self.sort_topological()

    
    def add_edge(self, from_node : Node, to_node : Node):
        """
        Adds an edge from 'from_node' to 'to_node'.
        """
        from_node.children.append(to_node) 
        to_node.parents.append(from_node)
        

    def sort_topological(self):
        """
        Topologically sorts the list of nodes.
        """
        graph = {node: node.parents for node in self.nodes}
        ts = TopologicalSorter(graph)
        self.nodes = list(ts.static_order())
        self.sorted = True

    def forward(self) -> Value:
        """
        Applies the forward pass in the graph.

        Returns:
            Value: a Value representing the output of the graph. In case there is an error Value.ERROR is returned.
        """
        if not self.sorted: 
            try:
                self.sort_topological()
            except Exception as e:
                print(f"Error during topological sort: {e}")
                return Value.ERROR

        outputs = []   
        for node in self.nodes:
            out = node.forward()
            if node in self.output_nodes:
                outputs.append(out)
        if len(outputs) == 0:
            return out
        return Value([out.value for out in outputs])
    
    def backward(self, grad_init: bool = True) -> List[Value]:
        """
        Computes the backward pass of the graph.

        Returns:
            List[Value]: list of gradients of the input nodes.
        """
        if not self.sorted:
            try:
                self.sort_topological()
            except Exception as e:
                print(f"Error during topological sort: {e}")
                return []
        outputs = []
        for node in reversed(self.nodes):
            if node in self.output_nodes and grad_init:
                node.upstream_grad = Value(1)
            out = node.backward()
            if node in self.input_nodes:
                outputs.append(out)
        return outputs
    
    def zero_grad(self):
        """
        Zeroes the gradients of all nodes.
        """
        for node in self.nodes:
            node.local_grad = Value.ZERO
            node.upstream_grad = Value.ZERO
