from tinynn import np
from tinynn.activations import Relu,Sigmoid,Softmax


class Dense:
    def __init__(self, input_size, n_neurons, layer_num, activation='relu'):
        self.weights = 0.01 * np.random.rand(input_size, n_neurons)
        self.biases = np.zeros(n_neurons)
        self.activation = activation
        self.inputs = None
        self.type = "Dense"
        self.layer_num = layer_num

    def __repr__(self):
        return f"Layer type: {self.type}, activation: {self.activation}, layer_num: {self.layer_num}"

    def forward(self, inputs):
        self.inputs = inputs
        forward_pass_outputs = np.dot(inputs, self.weights) + self.biases

        if self.activation == 'relu':
            self.activation = Relu(inputs=forward_pass_outputs)

        elif self.activation == 'sigmoid':
            self.activation = Sigmoid(inputs=forward_pass_outputs)

        elif self.activation == 'softmax':
            self.activation = Softmax(inputs=forward_pass_outputs)

        self.outputs = self.activation.outputs
        return self.outputs

    def backward(self, dvalues):
        self.activation.backward(dvalues)

        self.dweights = np.dot(self.inputs.T, self.activation.dinputs)
        self.dbiases = np.sum(self.activation.dinputs, axis=0, keepdims=True)
        self.dinputs = np.dot(self.activation.dinputs, self.weights.T)