from tinynn import np


class Relu:
    """
        Implements the Rectified Linear Unit (ReLU) activation function.
        ReLU is a popular activation function used in neural networks to introduce non-linearity.
        It returns 0 for negative inputs and the input value for positive inputs.

        Args:
            inputs (ndarray): Input values to the ReLU activation function.

        Attributes:
            inputs (ndarray): Input values to the ReLU activation function.
            outputs (ndarray): Output values after applying the ReLU activation.
            dinputs (ndarray): Gradient of the loss function with respect to the inputs.

        Methods:
            __init__(inputs): Initializes the ReLU activation function with the provided inputs.
            __repr__(): Returns a string representation of the ReLU activation function.
            backward(dvalues): Performs the backward pass for the ReLU activation function.

        """

    def __init__(self, inputs):
        """
        Initializes the ReLU activation function with the provided inputs.

        Args:
        inputs (ndarray): Input values to the ReLU activation function.
        """
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def __repr__(self):
        """
        Returns a string representation of the ReLU activation function.

        Returns:
        str: String representation of the ReLU activation function.
        """
        return "Relu Activation"

    def backward(self, dvalues):
        """
        Performs the backward pass for the ReLU activation function.

        The backward pass computes the gradient of the loss function with respect to the inputs,
        which is stored in the `dinputs` attribute.

        Args:
            dvalues (ndarray): Gradient of the loss function with respect to the outputs.

        """

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
