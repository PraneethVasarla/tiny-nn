from setuptools import setup, find_packages

setup(
    name='tinynn-py',
    version='0.1.1',
    packages=find_packages(),
    author='Praneeth Vasarla',
    description='A tiny neural network library.',
    install_requires=[
        "numpy"
    ],
    long_description="""
    # Tinynn

Tinynn is a lightweight neural network library built entirely in NumPy, designed to help developers understand the nitty-gritty details of neural networks. With its clean and concise code, Tinynn offers a great learning platform for those who want to deepen their understanding of the inner workings of neural networks.

Despite its simplicity, Tinynn has all the key features of a neural network library, including support for feedforward and recurrent networks, various activation functions, and common optimization algorithms. Whether you're a student, researcher, or machine learning enthusiast, Tinynn is an excellent tool for experimenting with neural network architectures and training algorithms.

## Key Features

- Lightweight and easy-to-understand implementation
- Support for feedforward and recurrent networks
- Various activation functions to choose from
- Common optimization algorithms
- Built entirely in NumPy for efficient numerical computations

## Installation

To install Tinynn, you can use pip:

```bash
pip install tinynn
```

## Usage
Here's a basic example demonstrating how to create a feedforward neural network using Tinynn:

```python
import tinynn as tn
import numpy as np
X = np.array([[1,2,3],
     [5,4,3],
     [2,3,4]])
y = np.array([0, 0, 1])


model = tn.models.Sequential()

model.add(tn.layers.Dense(3,64))
model.add(tn.layers.Dense(64,2,activation='softmax'))

model.compile_model(learning_rate=0.01, optimizer='adam') #Also available params: decay_rate,momentum(only for sgd) and optimizer = sgd,adagrad
model.fit(X,y,epochs=1000)
```
For more details, visit the github repo: https://github.com/PraneethVasarla/tiny-nn
    """
)
