from tinynn import np

#Optimizers
class StochasticGradientDescent:
    def __init__(self,layers,learning_rate=1.0):
        self.learning_rate = learning_rate
        self.layers = layers
    def update_parameters(self):
        for layer in self.layers:
            if layer.type != "Loss":
                layer.weights += -self.learning_rate * layer.dweights
                layer.biases += -self.learning_rate * layer.dbiases.reshape(layer.biases.shape)
                if layer.layer_num == 2:
                    pass
                    # print(layer)
                    # print("Weights:")
                    # print(layer.weights)
                    # print("")
                    # print("Bias")
                    # print(layer.biases)