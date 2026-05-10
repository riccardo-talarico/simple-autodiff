# Simple Autodiff

A smimple automatic differentiation engine built from scratch using NumPy.

This project implements:
- Computational graphs
- Backpropagation
- Basic neural network layers
- Basic Activation functions (ReLu, Softmax)
- Loss functions (Mean-squared-error, Cross-entropy)
- Simple training loop with mini-batching

The goal of the project is educational: replicating the backpropagation logic and implementation used in modern deep learning frameworks.

---

## Features

- Dynamic computational graphs
- Automatic gradient computation
- Linear layers
- ReLU activation
- Softmax + Cross Entropy
- Mean Squared Error loss
- SGD optimization
- He initialization
- Mini-batch training support

---

## Project Structure

```text
.
|
├──  autodiff/
|   │
|   ├── __init__.py
|   ├── computational_graph.py
|   ├── node.py
|   ├── value.py
|   ├── layers.py
|   ├── models.py
|   ├── losses.py
|   └── datasets.py
├── examples/ # Four Examples of usage on simple datasets
```

Example training scripts:
- Linear regression (`y = 2x + 1`)
- XOR classification
- 2D cluster classification
- Concentric circles classification

---

## Example

```python
from autodiff.models import Model
from autodiff.layers import Linear, ReLu

model = Model([
    Linear((2, 1),(16,1), init_mode='he'),
    ReLu((16,2)),
    Linear((16, 1),(1,1), init_mode='random')
])
```

---

## Implemented Concepts

### Automatic Differentiation
The engine constructs a computational graph during the forward pass and computes gradients through backward autodifferentiation.

### Neural Network Components
Implemented from scratch:
- Dense layers
- Activations
- Losses
- Parameter updates

### Initialization
Supports He initialization for ReLU-based networks.

---

## Results

The engine successfully trains:
- Linear regression models
- XOR neural networks
- Linear classifiers on synthetic blobs datasets
- Nonlinear classifiers on synthetic concentric circles datasets

---

## Installation

Clone the repository:

```bash
git clone https://github.com/riccardo-talarico/simple-autodiff.git
cd REPO
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dependencies

- numpy
- matplotlib
