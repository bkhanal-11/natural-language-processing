import numpy as np

from lstm_utils import *
from utils import *

# Generate test data
n_x = 10
m = 100
T_x = 20
n_y = 5
n_a = 50

learning_rate = 0.001

x = np.random.randn(n_x, m, T_x)
a0 = np.zeros((n_a, m))
da0 = np.zeros((n_a, m, 4))
y = np.random.randn(n_y, m, T_x)

# Initialize parameters
parameters = initialize_parameters_lstm(n_x, n_a, n_y)

# Forward Propagation
a, y_pred, c, caches = lstm_forward(x, a0, parameters)

# Compute Loss
loss = compute_loss(y_pred, y)

# Backward Propagation
grads = lstm_backward(da0, caches)

# Update Parameters
parameters = update_parameters_lstm(parameters, grads, learning_rate)

# Repeat for several epochs
num_epochs = 10
for i in range(num_epochs):
    # Forward Propagation
    a, y_pred, c, caches = lstm_forward(x, a0, parameters)

    # Compute Loss
    loss = compute_loss(y_pred, y)

    # Backward Propagation
    grads = lstm_backward(da0, caches)

    # Update Parameters
    parameters = update_parameters_lstm(parameters, grads, learning_rate)

    # Print Loss
    print("Epoch:", i, " Loss:", loss)
