import numpy as np

from sigmoid import Sigmoid


class Predict:
    def predict(self, Theta1, Theta2, X):
        # PREDICT predicts the label of an input given a trained neural network
        # p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
        # trained weights of a neural network (Theta1, Theta2)

        # turns 1D X array into 2D
        if X.ndim == 1:
            X = np.reshape(X, (-1, X.shape[0]))

        # Useful values
        m = X.shape[0]

        p = np.zeros((m, 1))
        s = Sigmoid()
        h1 = s.sigmoid(np.dot(np.column_stack((np.ones((m, 1)), X)), Theta1.T))
        h2 = s.sigmoid(np.dot(np.column_stack((np.ones((m, 1)), h1)), Theta2.T))

        p = np.argmax(h2, axis=1)

        return p + 1  # offsets python's zero notation since the 0th element is '10' and count of digits starts from 2nd index
