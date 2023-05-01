from tinynn import np

class Relu:
    def __init__(self,inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0,inputs)
    def __repr__(self):
        return "Relu Activation"
    def backward(self,dvalues):
        self.dinputs = self.outputs.copy()
        self.dinputs[dvalues<=0] = 0