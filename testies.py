import numpy as np

def sigmoid(x):
  
    sig = 1 / (1 + np.exp(-x))

    return sig

print(sigmoid(np.asarray([1, 2, 3, 4, 5])))