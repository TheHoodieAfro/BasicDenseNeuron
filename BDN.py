import numpy as np
import random as rng

"""Class based on the work showed in https://www.kdnuggets.com/2018/10/simple-neural-network-python.html"""
class BDN():
    
    """Initilize the weights and bias"""
    def __init__(self):
        
        self.w = rng.sample(range(-1, 1), (3,1))
        self.b = 0

    """Activation function"""
    def __activationFunction(self, z, f):

      return 1 / (1 + np.exp(-z))

    def predict(self, input):
        
        input = input.astype(float)
        z = np.dot(input, self.w) + self.b
        output = self.__activationFunction(z, 'sigmoid')
        return output, z

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