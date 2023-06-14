from tinynn import np
from tinynn.optimizers import StochasticGradientDescent,Adagrad,RMSProp,Adam
from tinynn.loss_functions import CategoricalCrossEntropy

class Sequential:
    """
    Represents a sequential neural network model.

    Attributes:
        layers (list): List of layers in the model.
        layers_count (int): Number of layers in the model.
        optimizer (object): Optimizer used for updating model parameters during training.

    Methods:
        add(layer_object): Adds a layer to the model.
        forward(inputs, labels): Performs forward propagation through the model.
        backward(y): Performs backward propagation through the model.
        compile_model(optimizer, loss, learning_rate, decay_rate, momentum, epsilon, rho, beta_1, beta_2): Compiles the model with the specified optimizer and loss function.
        fit(X, y, epochs): Trains the model on the given data.

    """
    def __init__(self):
        """
        Initializes the Sequential model.
        """
        self.layers = []
        self.layers_count = 0
        self.optimizer = None

    def add(self, layer_object):
        """
        Adds a layer to the model.

        Args:
            layer_object (object): Layer object to be added to the model.
        """
        self.layers_count += 1
        layer_object.layer_num = self.layers_count
        if self.layers_count > 1:
            layer_object.prev = self.layers[-1]
            self.layers[-1].next = layer_object
        self.layers.append(layer_object)

    def forward(self, inputs, labels):
        """
        Performs forward propagation through the model.

        Args:
            inputs (ndarray): Input data.
            labels (ndarray): True labels.
        """
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
        """
        Performs backward propagation through the model.

        Args:
            y (ndarray): True labels.
        """
        for layer in self.layers[::-1]:
            if layer.type == 'Loss':
                layer.backward(layer.prev.outputs, y)
            else:
                layer.backward(layer.next.dinputs)

    def compile_model(self, optimizer='sgd', loss='categorical_cross_entropy', learning_rate=0.01,decay_rate = None,momentum = None,epsilon = 1e-7,rho = 0.9,beta_1 = 0.9, beta_2 = 0.999):
        """
        Compiles the model with the specified optimizer and loss function.

        Args:
            optimizer (str): Optimizer to be used. Options: 'sgd', 'adagrad', 'rmsprop', 'adam'.
            loss (str): Loss function to be used. Options: 'categorical_cross_entropy'.
            learning_rate (float): Learning rate for the optimizer.
            decay_rate (float): Decay rate for learning rate decay (if applicable to the optimizer).
            momentum (float): Momentum value for the optimizer (if applicable).
            epsilon (float): Small value for numerical stability (if applicable to the optimizer).
            rho (float): Rho value for RMSProp optimizer (if applicable).
            beta_1 (float): Beta 1 value for Adam optimizer (if applicable).
            beta_2 (float): Beta 2 value for Adam optimizer (if applicable).
        """
        if loss == 'categorical_cross_entropy':
            self.add(CategoricalCrossEntropy())
        if optimizer == 'sgd':
            self.optimizer = StochasticGradientDescent(layers=self.layers, learning_rate=learning_rate,decay_rate=decay_rate,momentum=momentum)
        elif optimizer == 'adagrad':
            self.optimizer = Adagrad(layers=self.layers,learning_rate=learning_rate,decay_rate=decay_rate,epsilon=epsilon)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSProp(layers=self.layers,learning_rate=learning_rate,decay_rate=decay_rate,epsilon=epsilon,rho = rho)
        elif optimizer == 'adam':
            self.optimizer = Adam(layers=self.layers,learning_rate=learning_rate,decay_rate=decay_rate,epsilon=epsilon)

    def fit(self, X, y, epochs=10):
        """
        Trains the model on the given data.

        Args:
            X (ndarray): Input data.
            y (ndarray): True labels.
            epochs (int): Number of epochs to train the model.
        """
        if self.optimizer:
            for epoch in range(epochs):
                self.forward(X, y)
                self.backward(y)
                print(f"Epoch: {epoch + 1} | LR: {self.optimizer.current_learning_rate} | Loss: {self.loss}")
                self.optimizer.update_parameters()
        else:
            raise Exception("Model needs to be compiled first using model.compile_model()")