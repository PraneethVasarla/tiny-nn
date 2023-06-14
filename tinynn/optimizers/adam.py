from tinynn import np

class Adam:
    """
    Adam optimizer for updating model parameters during training.

    Attributes:
        learning_rate (float): Initial learning rate.
        current_learning_rate (float): Current learning rate.
        layers (list): List of layers in the model.
        decay_rate (float): Decay rate for learning rate decay.
        epochs (int): Number of epochs.
        epsilon (float): Small value for numerical stability.
        beta_1 (float): Exponential decay rate for the first moment estimates.
        beta_2 (float): Exponential decay rate for the second moment estimates.

    Methods:
        get_learning_rate(): Updates the current learning rate based on the decay rate and number of epochs.
        update_parameters(): Updates the parameters of each layer using the Adam algorithm.

    """
    def __init__(self, layers, learning_rate=1.0, decay_rate=None, epsilon=1e-7, beta_1=0.9,beta_2=0.999):
        """
        Initializes the Adam optimizer.

        Args:
            layers (list): List of layers in the model.
            learning_rate (float): Initial learning rate.
            decay_rate (float): Decay rate for learning rate decay.
            epsilon (float): Small value for numerical stability.
            beta_1 (float): Exponential decay rate for the first moment estimates.
            beta_2 (float): Exponential decay rate for the second moment estimates.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.layers = layers
        self.decay_rate = decay_rate
        self.epochs = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def get_learning_rate(self):
        """
        Updates the current learning rate based on the decay rate and number of epochs.
        lr = initial_lr/(1+decay_rate*epoch)
        """
        if self.decay_rate:
            self.current_learning_rate = self.learning_rate / (1 + self.decay_rate * self.epochs)

    def update_parameters(self):
        """
        Updates the parameters of each layer using the Adam algorithm.
        """
        for layer in self.layers:
            if layer.type != "Loss":
                self.get_learning_rate()
                if not hasattr(layer, "weight_cache"):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.weight_cache = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)
                    layer.bias_cache = np.zeros_like(layer.biases)

                layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
                layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

                weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1**(self.epochs+1))
                bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1**(self.epochs+1))

                layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
                layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2

                weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.epochs + 1))
                bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.epochs + 1))

                layer.weights += -self.current_learning_rate * weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+self.epsilon)
                layer.biases += -self.current_learning_rate * bias_momentums_corrected/(np.sqrt(bias_cache_corrected) + self.epsilon)

                self.epochs += 1