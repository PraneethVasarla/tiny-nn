from tinynn import np
from tinynn.activations import Relu,Sigmoid,Softmax


class Dense:
    """
    Implements a fully connected dense layer in a neural network.

    Args:
        input_size (int): Number of input features.
        n_neurons (int): Number of neurons in the layer.
        layer_num (int, optional): Layer number in the network. Defaults to None.
        activation (str, optional): Activation function to be applied. Defaults to 'relu'.

    Attributes:
        weights (ndarray): Weight matrix of the layer.
        biases (ndarray): Bias matrix of the layer.
        activation (str): Activation function to be applied.
        inputs (ndarray): Input values to the layer.
        type (str): Type of layer.
        layer_num (int): Layer number in the network.
        prev (object): Reference to the previous layer in the network.
        next (object): Reference to the next layer in the network.
        activation_func (object): Instance of the activation function applied.

    Methods:
        __init__(input_size, n_neurons, layer_num, activation): Initializes the Dense layer with the provided parameters.
        __repr__(): Returns a string representation of the Dense layer.
        forward(inputs): Performs forward propagation through the Dense layer.
        backward(dvalues): Performs backward propagation through the Dense layer.

    """
    def __init__(self, input_size, n_neurons, layer_num = None, activation='relu'):
        """
        Initializes the Dense layer with the provided parameters.

        Args:
            input_size (int): Number of input features.
            n_neurons (int): Number of neurons in the layer.
            layer_num (int, optional): Layer number in the network. Defaults to None.
            activation (str, optional): Activation function to be applied. Defaults to 'relu'.
        """
        self.weights = 0.01 * np.random.randn(input_size, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.activation = activation
        self.inputs = None
        self.type = "Dense"
        self.layer_num = layer_num
        self.prev = None
        self.next = None

    def __repr__(self):
        """
        Returns a string representation of the Dense layer.

        Returns:
            str: String representation of the Dense layer.
        """
        return f"Layer type: {self.type}, activation: {self.activation}, layer_num: {self.layer_num}"

    def forward(self, inputs):
        """
        Performs forward propagation through the Dense layer.

        Applies the activation function to the weighted sum of inputs and stores the outputs.

        Args:
            inputs (ndarray): Input values to the Dense layer.
        """
        self.inputs = inputs
        forward_pass_outputs = np.dot(inputs, self.weights) + self.biases

        if self.activation == 'relu':
            self.activation_func = Relu(inputs=forward_pass_outputs)

        elif self.activation == 'sigmoid':
            self.activation_func = Sigmoid(inputs=forward_pass_outputs)

        elif self.activation == 'softmax':
            self.activation_func = Softmax(inputs=forward_pass_outputs)

        self.outputs = self.activation_func.outputs

    def backward(self, dvalues):
        """
        Performs backward propagation through the Dense layer.

        Updates the gradients of the weights, biases, and inputs based on the gradient of the loss function.

        Args:
            dvalues (ndarray): Gradient of the loss function with respect to the outputs.
        """
        self.activation_func.backward(dvalues)
        self.dweights = np.dot(self.inputs.T, self.activation_func.dinputs)
        self.dbiases = np.sum(self.activation_func.dinputs, axis=0, keepdims=True)
        self.dinputs = np.dot(self.activation_func.dinputs, self.weights.T)