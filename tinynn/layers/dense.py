from tinynn import np
from tinynn.activations import Relu,Sigmoid,Softmax


class Dense:
    def __init__(self, input_size, n_neurons, layer_num = None, activation='relu'):
        self.weights = 0.01 * np.random.randn(input_size, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.activation = activation
        self.inputs = None
        self.type = "Dense"
        self.layer_num = layer_num
        self.prev = None
        self.next = None

    def __repr__(self):
        return f"Layer type: {self.type}, activation: {self.activation}, layer_num: {self.layer_num}"

    def forward(self, inputs):
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
        self.activation_func.backward(dvalues)
        self.dweights = np.dot(self.inputs.T, self.activation_func.dinputs)
        self.dbiases = np.sum(self.activation_func.dinputs, axis=0, keepdims=True)
        self.dinputs = np.dot(self.activation_func.dinputs, self.weights.T)