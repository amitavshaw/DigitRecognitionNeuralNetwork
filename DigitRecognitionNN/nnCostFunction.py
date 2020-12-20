import numpy as np

from sigmoid import Sigmoid

class NNCostFunction:
    """
     NNCOSTFUNCTION Implements the neural network cost function for a two layer
     neural network which performs classification
     [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels,X, y, lambda) 
     computes the cost and gradient of the neural network. The parameters for the neural network 
     are "unrolled" into the vector nn_params and need to be converted back into the weight matrices. 
     Also the back propagation algorithm and regularization are implemented.
    """
    def nnCostFunction(self, nn_params, input_layer_size, hidden_layer_size,
                       num_labels, X, y, lambda_reg):
        # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
        # for our 2 layer neural network
        Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                            (hidden_layer_size, input_layer_size + 1), order='F')

        Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                            (num_labels, hidden_layer_size + 1), order='F')

        # Setup some useful variables
        m = len(X)

        J = 0;
        Theta1_grad = np.zeros(Theta1.shape)
        Theta2_grad = np.zeros(Theta2.shape)

        # Part 1: Feedforward the neural network and return the cost in the
        #         variable J. The cost computed in main_script.py can be verified
        #
        # Part 2: Back-propagation algorithm to compute the gradients Theta1_grad and Theta2_grad.
        #
        # Part 3: Regularization with the cost function and gradients implemented.
        #         The gradients for the regularization are computed and then added to Theta1_grad
        #         and Theta2_grad from Part 2.
        #

        # add column of ones as bias unit from input layer to second layer
        X = np.column_stack((np.ones((m, 1)), X))  # = a1

        s = Sigmoid()
        # calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
        a2 = s.sigmoid(np.dot(X, Theta1.T))

        # add column of ones as bias unit from second layer to third layer
        a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))

        # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
        a3 = s.sigmoid(np.dot(a2, Theta2.T))

        # NON-REGULARIZED COST FUNCTION CALCULATION
        # Output labels as vectors containing only values 0 or 1
        labels = y
        # set y to be matrix of size m x k
        y = np.zeros((m, num_labels))
        # for every label, convert it into vector of 0s and a 1 in the appropriate position
        for i in range(m):
            y[i, labels[i] - 1] = 1

        # at this point, both a3 and y are m x k matrices, where m is the number of inputs
        # and k is the number of hypotheses. Given that the cost function is a sum
        # over m and k, loop over m and in each loop, sum over k by doing a sum over the row

        cost = 0
        for i in range(m):
            cost += np.sum(y[i] * np.log(a3[i]) + (1 - y[i]) * np.log(1 - a3[i]))

        J = -(1.0 / m) * cost

        # REGULARIZED COST FUNCTION

        sumOfTheta1 = np.sum(np.sum(Theta1[:, 1:] ** 2))
        sumOfTheta2 = np.sum(np.sum(Theta2[:, 1:] ** 2))

        J = J + ((lambda_reg / (2.0 * m)) * (sumOfTheta1 + sumOfTheta2))

        # BACK PROPAGATION
        bigDelta1 = 0
        bigDelta2 = 0

        # for each training example
        for t in range(m):
            # step 1: performing forward pass
            # set lowercase x to the t-th row of X
            x = X[t]
            # X already included column of ones 

            # calculating second layer as sigmoid( z2 ) where z2 = Theta1 * a1
            a2 = s.sigmoid(np.dot(x, Theta1.T))

            # adding column of ones as bias unit from second layer to third layer
            a2 = np.concatenate((np.array([1]), a2))
            # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
            a3 = s.sigmoid(np.dot(a2, Theta2.T))

            # step 2: for each output unit k in layer 3, set delta_{k}^{(3)}d
            delta3 = np.zeros((num_labels))

            # y_k indicates whether the current training example belongs to class k (y_k = 1),
            # or if it belongs to a different class (y_k = 0)
            for k in range(num_labels):
                y_k = y[t, k]
                delta3[k] = a3[k] - y_k

            # step 3: for the hidden layer l=2, set delta2 = Theta2' * delta3 .* sigmoidGradient(z2)
            sg = Sigmoid()
            delta2 = (np.dot(Theta2[:, 1:].T, delta3).T) * sg.sigmoidgradient(np.dot(x, Theta1.T))

            # step 4: accumulate gradient from this example
            bigDelta1 += np.outer(delta2, x)
            bigDelta2 += np.outer(delta3, a2)

        # step 5: obtaining gradient for neural net cost function by dividing the accumulated gradients by m
        Theta1_grad = bigDelta1 / m
        Theta2_grad = bigDelta2 / m

        # % REGULARIZATION FOR GRADIENT
        # only regularize for j >= 1, so skipping the first column
        Theta1_grad_unregularized = np.copy(Theta1_grad)
        Theta2_grad_unregularized = np.copy(Theta2_grad)
        Theta1_grad += (float(lambda_reg) / m) * Theta1
        Theta2_grad += (float(lambda_reg) / m) * Theta2
        Theta1_grad[:, 0] = Theta1_grad_unregularized[:, 0]
        Theta2_grad[:, 0] = Theta2_grad_unregularized[:, 0]

        # Unrolling gradients
        grad = np.concatenate(
            (Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))

        return J, grad