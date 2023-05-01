from tinynn import np

class Loss:
    def calculate(self, preds, labels):
        individual_losses = self.forward(preds, labels)
        total_loss = np.mean(individual_losses)
        if np.isnan(total_loss):
            print(individual_losses)
        return total_loss