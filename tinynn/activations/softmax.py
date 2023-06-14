from tinynn import np

class Softmax:
    """
    Implements the Softmax activation function.
    Softmax is commonly used as the output activation function in multi-class classification problems.
    It calculates the probabilities of each class based on the input values.

    Args:
        inputs (ndarray): Input values to the Softmax activation function.

    Attributes:
        outputs (ndarray): Output values after applying the Softmax activation.
        dinputs (ndarray): Gradient of the loss function with respect to the inputs.

    Methods:
        __init__(inputs): Initializes the Softmax activation function with the provided inputs.
        __repr__(): Returns a string representation of the Softmax activation function.
        backward(dvalues): Performs the backward pass for the Softmax activation function.

    """
    def __init__(self, inputs):
        """
        Initializes the Softmax activation function with the provided inputs.

        Args:
            inputs (ndarray): Input values to the Softmax activation function.
        """
        # We subtract each input with the sample max to avoid exploding value of exponentiation which may lead to
        # overflow error in case of large numbers
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values, axis=1,
                                           keepdims=True)  # These can also be referred to as probabilities
        self.outputs = probs

    def __repr__(self):
        """
        Returns a string representation of the Softmax activation function.

        Returns:
            str: String representation of the Softmax activation function.
        """
        return "Softmax Activation"

    def backward(self, dvalues):
        """
        Performs the backward pass for the Softmax activation function.

        The backward pass computes the gradient of the loss function with respect to the inputs,
        which is stored in the `dinputs` attribute.

        Args:
            dvalues (ndarray): Gradient of the loss function with respect to the outputs.
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)