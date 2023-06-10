from tinynn import np

class Adagrad:
    def __init__(self, layers, learning_rate=1.0, decay_rate=None, momentum=None):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.layers = layers
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.epochs = 0
        self.epsilon = 1e-7

    def get_learning_rate(self):
        if self.decay_rate:  # Implementing learning rate decay. lr = initial_lr/(1+decay_rate*epoch)
            self.current_learning_rate = self.learning_rate / (1 + self.decay_rate * self.epochs)
    def update_parameters(self):
        for layer in self.layers:
            if layer.type != "Loss":
                self.get_learning_rate()
                if not hasattr(layer, "weight_cache"):
                    layer.weight_cache = np.zeros_like(layer.weights)
                    layer.bias_cache = np.zeros_like(layer.biases)

                layer.weight_cache += layer.dweights**2
                layer.bias_cache += layer.dbiases**2

                layer.weights += -self.current_learning_rate * layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
                layer.biases += -self.current_learning_rate * layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)

                self.epochs += 1
