from tinynn import np
from tinynn.loss_functions import Loss

class CategoricalCrossEntropy(Loss):
    """
    Implements the Categorical Cross-Entropy loss function for multi-class classification problems.

    Attributes:
        type (str): Type of loss function.
        prev (object): Reference to the previous layer in the network.
        next (object): Reference to the next layer in the network.
        layer_num (int): Layer number in the network.

    Methods:
        __init__(): Initializes the CategoricalCrossEntropy loss function.
        forward(preds, labels): Performs forward propagation through the loss function.
        backward(dvalues, y_true): Performs backward propagation through the loss function.

    """
    def __init__(self):
        """
        Initializes the CategoricalCrossEntropy loss function.
        """
        self.type = "Loss"
        self.prev = None
        self.next = None
        self.layer_num = None

    def forward(self, preds, labels):
        """
        Performs forward propagation through the CategoricalCrossEntropy loss function.

        Calculates the negative log likelihoods based on the predicted probabilities and the true labels.

        Args:
            preds (ndarray): Predicted probabilities from the model.
            labels (ndarray): True labels.

        Returns:
            ndarray: Negative log likelihoods.
        """
        samples = len(preds)
        y_pred_clipped = np.clip(preds, 1e-7, 1 - 1e-7)  # clipping 0 to a very small number to avoid infinity
        if len(labels.shape) == 1:  # in case of scaler labels [0,0,1] -> 0 for 1st row, 0 for 2nd row etc.
            actual_probs = y_pred_clipped[range(samples), labels]  # indexing the actual predicted probabilities from the nn output. Imitates argmax in a way.

        elif len(labels.shape) == 2:  # in case of OHE labels [[0,1],[0,1],[1,0]]
            actual_probs = np.sum(y_pred_clipped * labels, axis=1)

        else:
            raise Exception("Labels shape unknown. Pass labels as either a shape of (n,) or (n,n)")

        negative_log_likelihoods = -np.log(actual_probs)

        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """
        Performs backward propagation through the CategoricalCrossEntropy loss function.

        Calculates the gradient of the loss with respect to the inputs.

        Args:
            dvalues (ndarray): Gradient of the loss function with respect to the outputs of the previous layer.
            y_true (ndarray): True labels.

        Returns:
            ndarray: Gradient of the loss function with respect to the inputs.
        """
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples