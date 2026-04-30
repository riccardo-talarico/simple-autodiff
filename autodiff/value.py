import numpy as np

ACCEPTED_TYPES = int | float | np.float64 | list | np.ndarray

class Value():

    ERROR = np.nan

    def __init__(self, value: ACCEPTED_TYPES):
        self.value = self._adjust_shape(value)
        self.shape = self.value.shape

    def _adjust_shape(self, value):
        _type = type(value)
        if _type == int or _type == float:
            return np.array([[np.float64(value)]])
        elif _type == np.float64:
            return np.array([[value]])
        elif _type == list:
            # In case it gets an empty list:
            if len(value) == 0:
                raise ValueError("Cannot pass an empty list as a value")
            
            # Check if it's list of lists
            self.shape = self._infer_shape(value)
            value = self._convert_to_float64(value)

            if len(self.shape) == 1: 
                value = [value]
            return np.array(value)
        
        elif _type == np.ndarray:
            if len(value.shape) == 0:
                value = np.array([[value]], dtype=np.float64)
            elif len(value.shape) == 1:
                value = np.array([value], dtype=np.float64) 
            return value
        else:
            raise ValueError(f"Unexpected type for the value. Accepted types are {ACCEPTED_TYPES}, got {type(value)}.")
    

    def _infer_shape(self, lis):
        """
        Recursively checks if a nested list has a uniform shape.
        Returns the shape as a tuple if valid. Raises ValueError if jagged or mixed.
        """
        # If it's not a list, shape is empty tuple
        if not isinstance(lis, list):
            return ()
        
        if len(lis) == 0:
            return (0,)
        
        reference_shape = self._infer_shape(lis[0])
        for index, item in enumerate(lis[1:], start=1):
            item_shape = self._infer_shape(item)
            if item_shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch detected- Element at index 0 has shape {reference_shape}, "
                    f"but element at index {index} has shape {item_shape}."
                )
                
        # The shape of the current list is its length + the shape of its children
        return (len(lis),) + reference_shape

    def _convert_to_float64(self, data):
        
        if type(data) != list:
            try:
                return np.float64(data)
            except Exception as e:
                print(f"Error during conversion of data into np.float64: {e}")
                exit(1)
        else:
            return [self._convert_to_float64(el) for el in data]

    def __str__(self):
        return f"{self.value}"
    
    def __add__(self, other):
        if other.shape != self.shape:
            raise ValueError(f"Cannot add values with different shapes: operand 1 has shape {self.shape}, operand 2 has {other.shape}")
        return Value(self.value+other.value)

    def __sub__(self, other):
        if other.shape != self.shape:
            raise ValueError(f"Cannot subtract values with different shapes: operand 1 has shape {self.shape}, operand 2 has {other.shape}")
        return Value(self.value-other.value)
    
    def __matmul__(self, other):
        return Value(self.value @ other.value)
    
    def __mult__(self, other):
        if other.shape != self.shape:
            raise ValueError(f"Cannot multiply-elementwise values with different shapes: operand 1 has shape {self.shape}, operand 2 has {other.shape}")
        return Value(self.value * other.value)
    
    def __eq__(self, other):
        return self.value == other.value

if __name__ == '__main__':
    v = Value(np.array([5]))
    print(v)
    v2 = Value([5])
    print(v2)
    v3 = v+v2
    print(v3)
