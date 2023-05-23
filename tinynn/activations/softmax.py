from tinynn import np

class Softmax:
    """Softmax is a combination of exponentiation and normalisation"""

    def __init__(self, inputs):
        # We subtract each input with the sample max to avoid exploding value of exponentiation which may lead to
        # overflow error in case of large numbers
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values, axis=1,
                                           keepdims=True)  # These can also be referred to as probabilities
        self.outputs = probs

    def __repr__(self):
        return "Softmax Activation"

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)