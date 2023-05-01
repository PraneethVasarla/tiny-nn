from tinynn import np
from tinynn.loss_functions import Loss

# Incomplete class...will visit later
class MeanSquaredError(Loss):
    def __init__(self, preds, labels):
        if len(labels.shape) == 1:  # in case of scaler labels [0,0,1] -> 0 for 1st row, 0 for 2nd row etc.
            actual_probs = preds[range(
                len(labels)), labels]  # indexing the actual predicted probabilities from the nn output. Imitates argmax in a way

        elif len(labels.shape) == 2:  # in case of OHE labels [[0,1],[0,1],[1,0]]
            actual_probs = np.sum(preds * labels, axis=1)
