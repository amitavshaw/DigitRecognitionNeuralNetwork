import numpy as np
from scipy.special import expit


class Sigmoid:
    """ Sigmoid Class contains the method to calculate the sigmoid function. 
        g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function evaluated at z.
        This should work regardless of z being a matrix or a vector.
    """

    def sigmoid(self, z):
        g = np.zeros(z.shape)
        # Sigmoid of each value of z (z can be a matrix, vector or scalar).
        g = expit(z)
        return g

    def sigmoidgradient(self, z):
        g = 1.0 / (1.0 + np.exp(-z))
        g = g * (1 - g)
        return g
