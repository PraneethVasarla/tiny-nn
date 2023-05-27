from tinynn import np
from tinynn.optimizers import StochasticGradientDescent
from tinynn.loss_functions import CategoricalCrossEntropy

class Sequential:
    def __init__(self):
        self.layers = []
        self.layers_count = 0
        self.optimizer = None

    def add(self, layer_object):
        self.layers_count += 1
        layer_object.layer_num = self.layers_count
        if self.layers_count > 1:
            layer_object.prev = self.layers[-1]
            self.layers[-1].next = layer_object
        self.layers.append(layer_object)

    def forward(self, inputs, labels):
        for layer in self.layers:
            if layer.type == "Loss":
                self.loss = layer.calculate(layer.prev.outputs, labels)
            else:
                if not layer.prev:
                    layer.forward(inputs)
                else:
                    layer.forward(layer.prev.outputs)
                    if not layer.next:
                        self.output = layer.outputs


    def backward(self, y):
        # backpropagation
        for layer in self.layers[::-1]:
            if layer.type == 'Loss':
                layer.backward(layer.prev.outputs, y)
            else:
                layer.backward(layer.next.dinputs)

    def compile_model(self, optimizer='sgd', loss='categorical_cross_entropy', learning_rate=0.01):
        if loss == 'categorical_cross_entropy':
            self.add(CategoricalCrossEntropy())
        if optimizer == 'sgd':
            self.optimizer = StochasticGradientDescent(layers=self.layers, learning_rate=learning_rate)

    def fit(self, X, y, epochs=10):
        if self.optimizer:
            for epoch in range(epochs):
                self.forward(X, y)
                self.backward(y)
                print(f"Epoch: {epoch + 1} | Loss: {self.loss}")
                self.optimizer.update_parameters()
        else:
            raise Exception("Model needs to be compiled first using model.compile_model()")