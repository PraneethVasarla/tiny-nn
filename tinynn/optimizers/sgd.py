from tinynn import np

class StochasticGradientDescent:
    """
    Stochastic Gradient Descent optimizer for updating model parameters during training.

    Attributes:
        learning_rate (float): Initial learning rate.
        current_learning_rate (float): Current learning rate.
        layers (list): List of layers in the model.
        decay_rate (float): Decay rate for learning rate decay.
        momentum (float): Momentum for accelerated optimization.
        epochs (int): Number of epochs.

    Methods:
        get_learning_rate(): Updates the current learning rate based on the decay rate and number of epochs.
        update_parameters(): Updates the parameters of each layer using the Stochastic Gradient Descent algorithm.

    """
    def __init__(self,layers,learning_rate=1.0,decay_rate = None,momentum = None):
        """
        Initializes the Stochastic Gradient Descent optimizer.

        Args:
            layers (list): List of layers in the model.
            learning_rate (float): Initial learning rate.
            decay_rate (float): Decay rate for learning rate decay.
            momentum (float): Momentum for accelerated optimization.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.layers = layers
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.epochs = 0

    def get_learning_rate(self):
        """
        Updates the current learning rate based on the decay rate and number of epochs.
        lr = initial_lr/(1+decay_rate*epoch)
        """
        if self.decay_rate:  # Implementing learning rate decay. lr = initial_lr/(1+decay_rate*epoch)
            self.current_learning_rate = self.learning_rate / (1 + self.decay_rate * self.epochs)
    def update_parameters(self):
        """
        Updates the parameters of each layer using the Stochastic Gradient Descent algorithm.
        """
        for layer in self.layers:
            if layer.type != "Loss":
                self.get_learning_rate()
                if self.momentum:
                    if not hasattr(layer,"weight_momentums"):
                        layer.weight_momentums = np.zeros_like(layer.weights)
                        layer.bias_momentums = np.zeros_like(layer.biases)
                    weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
                    bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases

                    layer.weights += weight_updates
                    layer.biases += bias_updates
                else:
                    layer.weights += -self.current_learning_rate * layer.dweights
                    layer.biases += -self.current_learning_rate * layer.dbiases
                self.epochs += 1
