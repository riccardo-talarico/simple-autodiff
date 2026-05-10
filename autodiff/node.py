from autodiff.value import Value
from typing import Callable, List

class Node(object):
    def __init__(self, requires_grad: bool=True):
        self.parents = []
        self.children = []
        self.shape = None

        self.requires_grad = requires_grad
        self.value: Value | None = None
        self.inputs = None

        self.local_grad : List[Value] | Value = []
        self.upstream_grad : Value = Value(0)
        self.inputs = None

    def get_inputs(self) -> List[Value] | None:
        """
        Gets the inputs from the parent nodes
        """
        if len(self.parents)==0:
            return self.inputs
        self.inputs = []
        for parent in self.parents:
            self.inputs.append(parent.value)

        return self.inputs
    
    def forward(self):
        pass

    def backward(self):
        """
        Updates the upstream gradient of the parents with the corresponding local gradient.
        """
        
        for i,parent in enumerate(self.parents):
            if parent.upstream_grad is Value.ZERO:
                parent.upstream_grad = self.local_grad[i]
            else:
                parent.upstream_grad = parent.upstream_grad + self.local_grad[i] 


class CustomNode(Node):
    def __init__(
            self, 
            requires_grad:bool, 
            forward_op: Callable[[Node,List[Value]],Value], 
            backward_op: Callable[[Node,List[Value], Value], List[Value]]
            ):
        self.forward_op = forward_op
        self.backward_op = backward_op
        super().__init__(requires_grad)
    
    def forward(self):
        inputs = self.get_inputs()
        self.value = self.forward_op(self,inputs) 
        return self.value
    
    def backward(self):
        if self.requires_grad:
            self.local_grad = self.backward_op(self,self.inputs, self.upstream_grad) 
        super().backward()
        return self.local_grad



class AddNode(Node):
    def forward(self):

        x,y = self.get_inputs()
        self.x = x
        self.y = y

        self.value = x+y
        return self.value
    
    def backward(self):
        # upstream_grad = dL/dz
        dx = self.upstream_grad
        dy = self.upstream_grad
        if self.requires_grad:
            self.local_grad = [dx, dy] 
        # updates the parents grad
        super().backward()
        return self.local_grad
    
class SubNode(Node):
    def forward(self):
        x,y = self.get_inputs()
        self.x = x
        self.y = y

        self.value = x-y
        return self.value
    
    def backward(self):
        # upstream_grad = dL/dz
        dx = self.upstream_grad
        dy = Value((-1)*self.upstream_grad.value)
        if self.requires_grad:
            self.local_grad = [dx, dy] 
        # updates the parents grad
        super().backward()
        return self.local_grad
    

class MultNode(Node):
    def forward(self):
        x,y = self.get_inputs()
        self.x = x
        self.y = y
        self.value = x*y
        return self.value
    
    def backward(self):
        # upstream_grad = dL/dz
        dx = self.y * self.upstream_grad # [dz/dx * dL/dz] 
        dy = self.x * self.upstream_grad # [dz/dy * dL/dz]
        if self.requires_grad:
            self.local_grad = [dx, dy]
        super().backward()
        return self.local_grad

class MatrixMultNode(Node):
    def forward(self):
        A,B = self.get_inputs()
        self.A = A
        self.B = B
        self.value = A @ B
        return self.value
    
    def backward(self):
        # upstream_grad = dL/dz
        dA = self.upstream_grad @ self.B.transpose() # [dz/dA * dL/dz] 
        dB = self.A.transpose() @ self.upstream_grad # [dz/dB * dL/dz]

        if self.requires_grad:
            self.local_grad = [dA, dB]
        super().backward()

        return self.local_grad
    
class DataNode(Node):
    def __init__(self, value : Value, requires_grad = True):
        super().__init__(requires_grad)
        self.value = value
    
    def backward(self):
        self.local_grad = self.upstream_grad if self.requires_grad else self.local_grad
        return self.local_grad

