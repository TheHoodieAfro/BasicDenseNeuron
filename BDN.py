from logging import raiseExceptions
from winsound import SND_ASYNC
import numpy as np
import random as rng

"""
Implementation of a basic dense neuron

Cristian Sanchez Pelaez
Samuel Satizabal
"""
class BDN:

  """
  Constructor
  """
  def __init__(self, type, activation = 'relu'):
    # Initilize the weights and bias

    self.type = type
    self.activation = activation

    self.activation_functions_reg = ['relu']
    self.activation_functions_class = []

    self.loss_functions_reg = []
    self.loss_functions_class = []

    if (self.type.lower() != 'regression'.lower() and self.type.lower() != 'classification'.lower()):
      raise Exception('The neuron must be either for regression or classification')

    elif (self.type.lower() == 'regression'.lower() and self.activation.lower() not in self.functions_reg):
      raise Exception(('For regression select between the functions in', self.functions_reg))

    elif (self.activation.lower() not in self.functions_class):
      raise Exception(('For classification select between the functions in', self.functions_class))

  """
  Private methods
  """
  def __activationFunction(self):
    # Activation function to calculate the predicted output

    # TODO: Add activation function

    return None

  """
  Public methods
  """
  def predict(self, input):
        
    input = input.astype(float)

    self.z = np.dot(input, self.w) + self.b

    output = self.__activationFunction()

    return output

  def fit(self):

    self.w = rng.sample(range(-1, 1), (len(input),1)) # TODO: Implement the train function

    return None



  def __minsq(y_predicted, y):
      squared_error = (y_predicted - y) ** 2
      sum_squared_error = np.sum(squared_error)
      mse = sum_squared_error / y.size
      return mse
    
  def __lossClassification(self, yp, y):
      loss = 0

      for i in range(len(y)):
        loss += (y[i]*(np.log(yp[i])+((1-y[i])*np.log(1-yp[i]))))
      
      return (-loss)

  def costFunctionREG(self, yp, y):
      return __minsq(yp, y)

  def costFunctionCLASS(self, yp, y):
      return __lossClassification(yp, y)*(1/len(y))

  def __dL(yp, y): #TODO sacar derivada de L reg(MSE) y de calss(lossClassification)
      return 2*yp-y*yp-yp*y
    
  def __dY(z):
      return z * (1 - z)

  def train(self, input, expected, i):
        
        self.w = rng.sample(range(-1, 1), (len(input),1))

        for j in range(i):

            prediction, z = self.predict(input)

            self.w = __dL(prediction, expected) + __dY(z) + input

            self.b = __dL(prediction, expected) + __dY(z) + 1