import numpy as np
import random as rng

"""Class based on the work showed in https://www.kdnuggets.com/2018/10/simple-neural-network-python.html"""
class BDN():
    
    """Initilize the weights and bias"""
    def __init__(self):
        
        self.w = rng.sample(range(-1, 1), (3,1))
        self.b = 0

    """Activation function"""
    def __activationFunction(self, x, f):

        if f == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        
        elif f == 'RELU':   #TODO add RELU function
            return x
        
        return x

    """Derivate for the weights"""
    def __weightD(self, x):
        return x * (1 - x)  #TODO add derivate function
    
    """Derivate for the bias"""
    def __biasD(self, x):
        return x    #TODO add derivate function

    def train(self, input, output, i):
        
        for j in range(i):

            currentOutput = self.think(input)

            error = output - currentOutput
            
            newW = np.dot(input.T, error * self.__weightD(output))

            self.w += adjustments

    def predict(self, input):
        
        input = input.astype(float)
        z = np.dot(input, self.w) + self.b
        output = self.__activationFunction(z, 'sigmoid')
        return output