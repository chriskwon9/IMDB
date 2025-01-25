import numpy as np


# Forward

np.random.seed(0)
class RecurrentNetwork(object):
    def __init__(self):
        self.hidden_state = np.zeros((1,3))
        
        self.W_hh = np.random.randn(3,3)
        self.W_xh = np.random.randn(4,3)
        self.W_hy = np.random.randn(3,4)
        self.Bh = np.random.randn(1,3)
        self.By = np.random.randn(1,4)

    def forward(self, x):
        self.hidden_state = np.tanh(np.dot(x, self.W_xh) + np.dot(self.hidden_state, self.W_hh) + self.Bh)
        return np.dot(self.hidden_state, self.W_hy) + self.By
    

input_vector = np.ones((1,4))
random_network = RecurrentNetwork()

print (random_network.forward(input_vector))



