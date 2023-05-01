from tinynn import np
from tinynn.loss_functions import Loss

class CategoricalCrossEntropy(Loss):
    """This calculates the negative log loss. This is actually -y*log(y') but due to the OHE in numpy it derives down
    to just -log(y')"""

    def __init__(self):
        self.type = "Loss"
        self.preds = None

    def forward(self, preds, labels):
        self.preds = preds
        preds = np.clip(preds, 1e-09, 1 - 1e-09)  # clipping 0 to a very small number to avoid infinity
        if len(labels.shape) == 1:  # in case of scaler labels [0,0,1] -> 0 for 1st row, 0 for 2nd row etc.
            actual_probs = preds[range(
                len(labels)), labels]  # indexing the actual predicted probabilities from the nn output. Imitates argmax in a way.

        elif len(labels.shape) == 2:  # in case of OHE labels [[0,1],[0,1],[1,0]]
            actual_probs = np.sum(preds * labels, axis=1)

        else:
            raise Exception("Labels shape unknown. Pass labels as either a shape of (n,) or (n,n)")

        negative_log_likelihoods = -np.log(actual_probs)
        if np.isnan(negative_log_likelihoods[0]):
            print(f"ActualProbs: {actual_probs}")
            print(f"Preds: {preds}")
            print(f"Preds original: {self.preds}")

        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples