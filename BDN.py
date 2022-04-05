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
  def __init__(self, type, activation = 'relu', loss = 'mse'):
    # Initilize the weights and bias

    self.type = type
    self.activation = activation
    self.loss = loss

    self.activation_functions_reg = ['relu']
    self.activation_functions_class = ['sigmoid']

    self.loss_functions_reg = ['mse']
    self.loss_functions_class = []

    if (self.type.lower() != 'regression'.lower() and self.type.lower() != 'classification'.lower()):
      raise Exception('The neuron must be either for regression or classification')

    elif (self.type.lower() == 'regression'.lower() and self.activation.lower() not in self.activation_functions_reg and self.loss.lower() not in self.loss_functions_reg):
      raise Exception(('For regression select between the activation functions in', self.activation_functions_reg, 'and loss functions in', self.loss_functions_reg))

    elif (self.activation.lower() not in self.activation_functions_class and self.loss.lower() not in self.loss_functions_class):
      raise Exception(('For classification select between the functions in', self.activation_functions_class, ' and loss function in', self.loss_functions_class))

  """
  Private methods
  """
  def activation_function(self):
    # Activation function to calculate the predicted output
    
    if (self.activation.lower() == 'relu'):
      return np.maximum(self.z, 0)

    elif (self.activation.lower() == 'sigmoid'):
      return 1 / (1 + np.exp(-self.z))

    return None

  def loss_function(self, yp, y):
    # Loss function to calculate loss of precision in prediction

    if (self.loss.lower() == 'mse'):
      se = (yp - y) ** 2
      sse = np.sum(se)
      mse = sse / y.size

      return mse

    elif (self.loss.lower() == 'bce'):
      yp = np.clip(yp, 1e-7, 1 - 1e-7)
      t1 = (1-y) * np.log(1-yp + 1e-7)
      t2 = y * np.log(yp + 1e-7)
      return -np.mean(t1+t2, axis=0)

    return None

  def dY(self):
    # Derivate of the activation function

    if (self.activation.lower() == 'relu'):
      return 0

    elif (self.activation.lower() == 'sigmoid'):
      return 0

    return 0

  def dL(self, yp, y):
    # Derivate of the loss function

    if (self.loss.lower() == 'mse'):
      return 0

    elif (self.loss.lower() == 'bce'):
      return 0

    return 0

  """
  Public methods
  """
  def predict(self, input):
    # Prediction function based on the trainning
        
    input = input.astype(float)

    self.z = np.dot(input, self.w) + self.b

    output = self.activation_function()

    return output
    
  def fit(self, input, output, i):
    # Trainning function

    self.w = rng.sample(range(-1, 1), (len(input),1))

    for j in range(i):

            prediction = self.predict(input)

            loss = self.loss_function(prediction, output)

            self.w = self.dL(prediction, output) + self.dY(self.z) + input

            self.b = self.dL(prediction, output) + self.dY(self.z) + 1

    return None



  """def __dL(yp, y): #TODO sacar derivada de L reg(MSE) y de calss(lossClassification)
      return 2*yp-y*yp-yp*y
    
  def __dY(z):
      return z * (1 - z)

  def train(self, input, expected, i):
        
        self.w = rng.sample(range(-1, 1), (len(input),1))

        for j in range(i):

            prediction, z = self.predict(input)

            self.w = __dL(prediction, expected) + __dY(z) + input

            self.b = __dL(prediction, expected) + __dY(z) + 1"""