from tinynn import np

class Loss:
    """
    Base class for loss functions in a neural network.

    Methods:
        calculate(preds, labels): Calculates the overall loss for a batch of predictions and labels.

    """
    def calculate(self, preds, labels):
        """
        Calculates the overall loss for a batch of predictions and labels.

        Args:
            preds (ndarray): Predicted values from the model.
            labels (ndarray): True labels.

        Returns:
            float: The overall loss value.
        """
        individual_losses = self.forward(preds, labels)
        total_loss = np.mean(individual_losses)
        return total_loss