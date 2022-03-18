import numpy as np
import random as rng

#Inicializa los pesos con valores random
def __initializeWeights(size):

    w = np.array(rng.sample(range(-10, 10), size))

    return w

#Funcion de inicializacion
def __RELU(z):
	return max(0.0, z)

#Neurona
def BDN(x):

    w = __initializeWeights(len(x))
    b = 0

    z = w*x+b

    y = __RELU(z)

    print(y)