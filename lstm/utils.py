import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

import numpy as np

def initialize_parameters_lstm(n_x, n_a, n_y):
    """
    Initialize parameters for a 2-layer LSTM network.
    
    Arguments:
    n_x -- size of the input layer
    n_a -- size of the hidden state
    n_y -- size of the output layer
    
    Returns:
    parameters -- a dictionary of initialized parameters
    """
    np.random.seed(1)
    
    # forget gate weights
    Wf = np.random.randn(n_a, n_a + n_x) * 0.01
    bf = np.zeros((n_a, 1))
    
    # update gate weights
    Wi = np.random.randn(n_a, n_a + n_x) * 0.01
    bi = np.zeros((n_a, 1))
    
    # candidate value weights
    Wc = np.random.randn(n_a, n_a + n_x) * 0.01
    bc = np.zeros((n_a, 1))
    
    # output gate weights
    Wo = np.random.randn(n_a, n_a + n_x) * 0.01
    bo = np.zeros((n_a, 1))
    
    # prediction weights
    Wy = np.random.randn(n_y, n_a) * 0.01
    by = np.zeros((n_y, 1))
    
    # create dictionary of parameters
    parameters = {"Wf": Wf, "bf": bf, "Wi": Wi, "bi": bi, "Wc": Wc, "bc": bc,
                  "Wo": Wo, "bo": bo, "Wy": Wy, "by": by}
    
    return parameters

def compute_loss(predictions, targets):
    """
    Compute the average loss between the predicted values and the actual values.

    Arguments:
    predictions -- numpy array of shape (m, T_y, n_y), predicted output sequence
    targets -- numpy array of shape (m, T_y, n_y), actual output sequence

    Returns:
    loss -- float, average loss between the predicted values and the actual values
    """
    # Flatten the predictions and targets
    predictions = predictions.reshape(-1, predictions.shape[2])
    targets = targets.reshape(-1, targets.shape[2])

    # Calculate the cross-entropy loss
    loss = np.mean(-np.sum(targets * np.log(predictions + 1e-8), axis=1))

    return loss

def update_parameters_lstm(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing the parameters
    grads -- python dictionary containing the gradients, output of lstm_backward
    learning_rate -- learning rate used in the update rule

    Returns:
    parameters -- python dictionary containing the updated parameters
    """

    # Retrieve parameters
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Retrieve gradients
    dWf = grads["dWf"]
    dbf = grads["dbf"]
    dWi = grads["dWi"]
    dbi = grads["dbi"]
    dWc = grads["dWc"]
    dbc = grads["dbc"]
    dWo = grads["dWo"]
    dbo = grads["dbo"]

    # Update parameters
    Wf = Wf - learning_rate * dWf
    bf = bf - learning_rate * dbf
    Wi = Wi - learning_rate * dWi
    bi = bi - learning_rate * dbi
    Wc = Wc - learning_rate * dWc
    bc = bc - learning_rate * dbc
    Wo = Wo - learning_rate * dWo
    bo = bo - learning_rate * dbo

    # Store updated parameters
    parameters = {"Wf": Wf,
                  "bf": bf,
                  "Wi": Wi,
                  "bi": bi,
                  "Wc": Wc,
                  "bc": bc,
                  "Wo": Wo,
                  "bo": bo,
                  "Wy": Wy,
                  "by": by}

    return parameters
