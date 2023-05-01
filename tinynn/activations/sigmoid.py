from tinynn import np

class Sigmoid:
    def __init__(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
    def __repr__(self):
        return "Sigmoid Activation"