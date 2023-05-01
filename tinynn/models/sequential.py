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
        self.layers.append(layer_object)

    def forward(self, inputs, labels):
        output = inputs
        for layer in self.layers:
            if layer.type != "Loss":
                output = layer.forward(output)
            else:
                self.loss = layer.calculate(output, labels)

        self.output = output
        # if layer.type == 'Loss':
        # print(f"Loss: {self.output}")

    def backward(self, y):
        # backpropagation
        for index, layer in enumerate(self.layers[::-1]):
            if layer.type == 'Loss':
                layer.backward(self.layers[-2].outputs, y)
            else:
                layer.backward(self.layers[-abs(index)].dinputs)

    def compile_model(self, optimizer='sgd', loss='categorical_cross_entropy', learning_rate=0.01):
        if loss == 'categorical_cross_entropy':
            self.layers.append(CategoricalCrossEntropy())
        if optimizer == 'sgd':
            self.optimizer = StochasticGradientDescent(layers=self.layers, learning_rate=learning_rate)

    def fit(self, X, y, epochs=10):
        if self.optimizer:
            for epoch in range(epochs):
                self.forward(X, y)
                self.backward(y)
                print(f"Epoch: {epoch + 1} | Loss: {self.loss}")
                self.optimizer.update_parameters()
                # print(self.loss == "nan")
                if np.isnan(self.loss):
                    break
        else:
            raise Exception("Model needs to be compiled first using model.compile_model()")