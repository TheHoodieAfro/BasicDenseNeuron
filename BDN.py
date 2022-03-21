import numpy as np
import random as rng

"""Class based on the work showed in https://www.kdnuggets.com/2018/10/simple-neural-network-python.html"""
class BDN():
    
    """Initilize the weights and bias"""
    def __init__(self):
        
        self.w = rng.sample(range(-1, 1), (3,1))

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

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output