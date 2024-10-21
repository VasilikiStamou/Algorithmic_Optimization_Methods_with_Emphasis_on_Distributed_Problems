import numpy as np
import matplotlib.pyplot as plt

# Define the cost function and its gradient
def cost_function(x, A):
    return x.T @ A @ x
def gradient(x, A):
    return 2 * A @ x

# Gradient Descent Algorithm
def gradient_descent(A, x0, step_size, tol=1e-5, max_iter=1000):
    x = x0.astype(float)  # Ensure x is of float type
    cost_history = [cost_function(x, A)]
    x_history = [x.copy()]

    for i in range(max_iter):
        grad = gradient(x, A)
        x -= step_size * grad
        cost_history.append(cost_function(x, A))
        x_history.append(x.copy())

        # Check for convergence
        if cost_history[-1] < tol:
            break

    return np.array(x_history), np.array(cost_history), i + 1

# Backtracking line search
def backtracking_line_search(x, grad, A, alpha=1, beta=0.9, c=0.25):
    t = alpha
    while cost_function(x - t * grad, A) > cost_function(x, A) - c * t * grad.T @ grad:
        t *= beta
    return t

# Gradient Descent with Backtracking Line Search
def gradient_descent_backtracking(A, x0, tol=1e-5, max_iter=1000, alpha=1, beta=0.9, c=0.25):
    x = x0.astype(float)
    num_iters = 0

    for _ in range(max_iter):
        grad = gradient(x, A)
        step_size = backtracking_line_search(x, grad, A, alpha, beta, c)
        x -= step_size * grad
        num_iters += 1

        if np.linalg.norm(grad) < tol:
            break

    return num_iters

# Exact line search
def exact_line_search(A, grad):
    numerator = grad.T @ grad
    denominator = 2 * grad.T @ A @ grad
    return numerator / denominator

# Gradient Descent with Exact Line Search
def gradient_descent_exact(A, x0, tol=1e-5, max_iter=1000):
    x = x0.astype(float)
    num_iters = 0

    for _ in range(max_iter):
        grad = gradient(x, A)
        step_size = exact_line_search(A, grad)
        x -= step_size * grad
        num_iters += 1

        if np.linalg.norm(grad) < tol:
            break

    return num_iters