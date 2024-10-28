import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Args:
        y: np.array of shape (n, ) where n is the number of samples
        tx: np.array of shape (n, d) where d is the number of features
        initial_w: np.array of shape (d, ) where d is the number of features
        max_iters: int, number of iterations
        gamma: float, learning rate

    Returns:
        w: np.array of shape (d, ) where d is the number of features
        loss: float, the mean squared error
    """
    w = initial_w
    for n_iter in range(max_iters):
        error = y - tx @ w
        grad = -tx.T @ error / len(y)
        w = w - gamma * grad
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Args:
        y: np.array of shape (n, ) where n is the number of samples
        tx: np.array of shape (n, d) where d is the number of features
        initial_w: np.array of shape (d, ) where d is the number of features
        max_iters: int, number of iterations
        gamma: float, learning rate

    Returns:
        w: np.array of shape (d, ) where d is the number of features
        loss: float, the mean squared error
    """
    w = initial_w
    for n_iter in range(max_iters):
        i = np.random.randint(len(y))
        error = y[i] - tx[i] @ w
        grad = -tx[i].T * error
        w = w - gamma * grad
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


def least_squares(y, tx):
    """
    Args:
        y: np.array of shape (n, ) where n is the number of samples
        tx: np.array of shape (n, d) where d is the number of features

    Returns:
        w: np.array of shape (d, ) where d is the number of features
        loss: float, the mean squared error
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Args:
        y: np.array of shape (n, ) where n is the number of samples
        tx: np.array of shape (n, d) where d is the number of features
        lambda_: float, regularization parameter

    Returns:
        w: np.array of shape (d, ) where d is the number of features
        loss: float, the mean squared error
    """
    n, d = tx.shape
    lambda_prime = 2 * n * lambda_
    a = tx.T @ tx + lambda_prime * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Args:
        y: np.array of shape (n, ) where n is the number of samples
        tx: np.array of shape (n, d) where d is the number of features
        initial_w: np.array of shape (d, ) where d is the number of features
        max_iters: int, number of iterations
        gamma: float, learning rate

    Returns:
        w: np.array of shape (d, ) where d is the number of features
        loss: float, the logistic loss
    """
    w = initial_w

    for n_iter in range(max_iters):
        pred = tx @ w
        sigmoid = 1 / (1 + np.exp(-pred))
        error = sigmoid - y
        grad = tx.T @ error / len(y)
        w = w - gamma * grad

    sigmoid = 1 / (1 + np.exp(-tx @ w))
    loss = -np.mean(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Args:
        y: np.array of shape (n, ) where n is the number of samples
        tx: np.array of shape (n, d) where d is the number of features
        lambda_: float, regularization parameter
        initial_w: np.array of shape (d, ) where d is the number of features
        max_iters: int, number of iterations
        gamma: float, learning rate

    Returns:
        w: np.array of shape (d, ) where d is the number of features
        loss: float, the regularized logistic loss
    """
    w = initial_w

    for n_iter in range(max_iters):
        pred = tx @ w
        sigmoid = 1 / (1 + np.exp(-pred))
        error = sigmoid - y
        grad = tx.T @ error / len(y) + 2 * lambda_ * w
        w = w - gamma * grad

    sigmoid = 1 / (1 + np.exp(-tx @ w))
    loss = -np.mean(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
    return w, loss
