import numpy as np

class DebugInitializeWeights:
    """Initialize the weights of a layer with fan_in incoming connections and fan_out outgoing connections.
    W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights of a layer with fan_in incoming 
    connections and fan_out outgoing connections using a fix set of values.
    W should be set to a matrix of size(1 + fan_in, fan_out) as the first row of W handles the "bias" terms"""

    def debugInitializeWeights(self, fan_out, fan_in):
        # Set W to zeros
        W = np.zeros((fan_out, 1 + fan_in))

        # Initialize W using "sin", this ensures that W is always of the same
        # values and will be useful for debugging
        W = np.reshape(np.sin(range(W.size)), W.shape) / 10

        return W
