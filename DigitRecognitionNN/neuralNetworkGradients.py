import numpy as np

from debugInitializeWeights import DebugInitializeWeights
from nnCostFunction import NNCostFunction
from decimal import Decimal

class NeuralNetworkGradients:
    """CHECKNNGRADIENTS(lambda_reg) Creates a small neural network to check the back-propagation gradients,
    it will output the analytical gradients produced by back-propagation code and the numerical gradients (computed using computeNumericalGradient). These two gradient computations should result in very similar values."""

    def computeNumericalGradient(self, J, theta):
        # COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
        # and gives a numerical estimate of the gradient.
        # numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
        # gradient of the function J around theta. Calling y = J(theta) should
        # return the function value at theta.
        # The following code implements numerical gradient checking, and
        # returns the numerical gradient.It sets numgrad(i) to the partial derivative of J with respect to the
        # i-th input argument, evaluated at theta.

        numgrad = np.zeros(theta.shape)
        perturb = np.zeros(theta.shape)
        e = 1e-4

        for p in range(theta.size):
            # Set perturbation vector
            perturb.reshape(perturb.size, order="F")[p] = e
            loss1, _ = J(theta - perturb)
            loss2, _ = J(theta + perturb)
            # Compute Numerical Gradient
            numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2 * e)
            perturb.reshape(perturb.size, order="F")[p] = 0

        return numgrad

    def checkNNGradients(self, lambda_reg=0):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 5

        # Generate some 'random' test data
        diw = DebugInitializeWeights()
        Theta1 = diw.debugInitializeWeights(hidden_layer_size, input_layer_size)
        Theta2 = diw.debugInitializeWeights(num_labels, hidden_layer_size)
        # Using debugInitializeWeights to generate X
        X = diw.debugInitializeWeights(m, input_layer_size - 1)
        y = 1 + np.mod(range(m), num_labels).T

        # Unroll parameters
        nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

        nncf = NNCostFunction()

        def cost_func(p):
            return nncf.nnCostFunction(p, input_layer_size, hidden_layer_size,
                                       num_labels, X, y, lambda_reg)

        _, grad = cost_func(nn_params)
        numgrad = self.computeNumericalGradient(cost_func, nn_params)

        # The two columns should be very similar.
        # code from http://stackoverflow.com/a/27663954/583834
        fmt = '{:<25}{}'
        print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
        for numerical, analytical in zip(numgrad, grad):
            print(fmt.format(numerical, analytical))

        print('The above two columns should be very similar.\n' \
              '(Left Col.: Numerical Gradient, Right Col.: Analytical Gradient)')

        # Evaluating the norm of the difference between two solutions.  
        # If the implementation is correct, and assuming EPSILON = 0.0001
        # in computeNumericalGradient method, then diff below should be less than 1e-9
        diff = Decimal(np.linalg.norm(numgrad - grad)) / Decimal(np.linalg.norm(numgrad + grad))

        print('If back-propagation implementation is correct, then \n'
              'the relative difference will be small (less than 1e-9). \n'
              '\nRelative Difference: {:.10E}'.format(diff))


