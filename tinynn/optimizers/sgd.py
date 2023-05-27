from tinynn import np

class StochasticGradientDescent:
    def __init__(self,layers,learning_rate=1.0,decay_rate = None):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.layers = layers
        self.decay_rate = decay_rate
        self.epochs = 0
    def update_parameters(self):
        for layer in self.layers:
            if layer.type != "Loss":
                if self.decay_rate: # Implementing learning rate decay. lr = initial_lr/(1+decay_rate*epoch)
                    self.current_learning_rate = self.learning_rate/(1 + self.decay_rate * self.epochs)
                layer.weights += -self.current_learning_rate * layer.dweights
                layer.biases += -self.current_learning_rate * layer.dbiases
                self.epochs += 1
