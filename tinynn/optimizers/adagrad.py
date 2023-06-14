from tinynn import np

class Adagrad:
    """
        Adagrad optimizer for updating model parameters during training.

        Attributes:
            learning_rate (float): Initial learning rate.
            current_learning_rate (float): Current learning rate.
            layers (list): List of layers in the model.
            decay_rate (float): Decay rate for learning rate decay.
            epochs (int): Number of epochs.
            epsilon (float): Small value for numerical stability.

        Methods:
            get_learning_rate(): Updates the current learning rate based on the decay rate and number of epochs.
            update_parameters(): Updates the parameters of each layer using the Adagrad algorithm.

        """
    def __init__(self, layers, learning_rate=1.0, decay_rate=None, epsilon = 1e-7):
        """
        Initializes the Adagrad optimizer.

        Args:
            layers (list): List of layers in the model.
            learning_rate (float): Initial learning rate.
            decay_rate (float): Decay rate for learning rate decay.
            epsilon (float): Small value for numerical stability.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.layers = layers
        self.decay_rate = decay_rate
        self.epochs = 0
        self.epsilon = epsilon

    def get_learning_rate(self):
        """
        Updates the current learning rate based on the decay rate and number of epochs.
        lr = initial_lr/(1+decay_rate*epoch)
        """
        if self.decay_rate:
            self.current_learning_rate = self.learning_rate / (1 + self.decay_rate * self.epochs)
    def update_parameters(self):
        """
        Updates the parameters of each layer using the Adagrad algorithm.
        """
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
