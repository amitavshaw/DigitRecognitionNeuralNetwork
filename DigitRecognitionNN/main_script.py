# Neural Network implementation and predictions
import sys

import numpy as np
import scipy.io
from scipy.optimize import minimize

from displayData import DisplayData
from neuralNetworkGradients import NeuralNetworkGradients
from nnCostFunction import NNCostFunction
from predict import Predict

# Setup parameters
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10
# (mapped "0" to label 10)

# =========== Loading and Visualizing Data ================================
#  loading and visualizing the dataset. 
#  Dataset contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('training_set.mat')

X = mat["X"]
y = mat["y"]

m = X.shape[0]

# changes the dimension from (m,1) to (m,) for effective minimization
y = y.flatten()

# Randomly selecting 100 data points for dataset to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]
dd = DisplayData()
dd.displayData(sel)

# ================ Loading Parameters ================
# pre-initialized neural network parameters are loaded.

print('Loading Saved Neural Network Parameters ...')

# Loads the weights into variables Theta1 and Theta2
mat = scipy.io.loadmat('NNWeights.mat')
Theta1 = mat["Theta1"]
Theta2 = mat["Theta2"]

nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

# ================================== Compute Cost (Feedforward) ==============================================
#   Implementing the feedforward to compute the cost. The calculated cost is checked against a value provided. 
#   It is to verify the cost for fixed debugging parameters. This does not include regularization terms

print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lambda_reg = 0
nncf = NNCostFunction()
J, _ = nncf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                           num_labels, X, y, lambda_reg)

print('Training Set Accuracy: {:f}\n(this value should be about 0.287629)'.format(J))

# =============================== Implementing Regularization =================================
#  After cost function implementation is done, the regularization with the cost is implemented.
#

print('Checking Cost Function with Regularization...')

# Weight regularization parameter (we set this to 1 here).
lambda_reg = 1

J, _ = nncf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                           num_labels, X, y, lambda_reg)

print('Cost at parameters (loaded from NNWeights): {:f}\n(this value should be about 0.383770)'.format(J))

# ========================================== Implement Backpropagation ===========================================
#  backpropagation algorithm for the neural network must be implemented. nnCostFunction.py returns the partial
#  derivatives of the parameters.
#
print('Checking Backpropagation... ')

#  Check gradients by running checkNNGradients
cnng = NeuralNetworkGradients()
cnng.checkNNGradients()

# ================================== Implement Regularization =====================================================
#  Once backpropagation implementation is done, the regularization with the cost and gradient must be incorporated.
#

print('\nChecking Backpropagation with Regularization ... \n')

#  Check gradients by running checkNNGradients
lambda_reg = 3
cnng = NeuralNetworkGradients()
cnng.checkNNGradients(lambda_reg)

# Also output the costFunction debugging values
debug_J, _ = nncf.nnCostFunction(nn_params, input_layer_size,
                                 hidden_layer_size, num_labels, X, y, lambda_reg)

print('\n\nCost at (fixed) debugging parameters (w/ lambda_reg = 3): {:f} '
      '\n(this value should be about 0.576051)\n\n'.format(debug_J))

# =================== Training NN =======================================
#  scipy.optimize.minimize is used as optimizer here
# 
print('Training Neural Network...')

maxiter = 20
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
nncf = NNCostFunction()
results = minimize(nncf.nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter': maxiter},
                   method="L-BFGS-B", jac=True)

nn_params = results["x"]

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, input_layer_size + 1), order='F')

Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                    (num_labels, hidden_layer_size + 1), order='F')

pr = Predict()
pred = pr.predict(Theta1, Theta2, X)

print('Training Set Accuracy: {:f}'.format((np.mean(pred == y) * 100)))

input('Program paused. Press enter to continue.\n')

#  Checking the examples one at a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(m)

for i in range(m):
    # Display
    print('Displaying Example Image')
    dd.displayData(X[rp[i], :])

    pred = pr.predict(Theta1, Theta2, X[rp[i], :])
    print('Neural Network Prediction: {:d} (digit {:d})'.format(pred[0], (pred % 10)[0]))

    keyinput = input('Program paused. Press enter to continue or type \'q\' to exit\n')
    if keyinput == "q":
        sys.exit(0)
