import random
import numpy as np
import matplotlib.pyplot as plt

def squared_cost(x, y, w, b):  # for simple example
    pred = x * w + b
    return (pred - y) ** 2


def compute_mean_squared_cost_all(X, Y, w, b):
    cost = 0
    for i in range(len(X)):
        cost += squared_cost(X[i], Y[i], w, b)

    cost /= 2 * len(X)
    return cost

def compute_derivative_cost(X, Y, w, b):
    cost_w = 0
    cost_b = 0
    for i in range(len(X)):
        cost_w += (X[i] * w + b - Y[i]) * X[i]
        cost_b += (X[i] * w + b - Y[i])
    cost_w /= len(X)
    cost_b /= len(X)
    return cost_w, cost_b


def gradient_descent(initial_w, initial_b, n_iterations, X, Y, alpha):
    cost_history = []
    w, b = initial_w, initial_b

    for i in range(n_iterations):
        # calculate new w and b
        cost_w, cost_b = compute_derivative_cost(X, Y, w, b)
        w = w - alpha * cost_w
        b = b - alpha * cost_b

        # save and compare cost
        new_cost = compute_mean_squared_cost_all(X, Y, w, b)
        cost_history.append(new_cost)
        if i > 2 and abs(new_cost-cost_history[i-1]) < 0.01:
            break
        # update w and b
    return w, b,cost_history


x = np.array([5,10,15,25,30,50])
y = np.array([6,11,20,20,35,55])

plt.plot(x,y,'*')

w,b,history = gradient_descent(2,5,1000,x,y,0.0001)
print(len(history))
plt.plot(x,x*w+b)
plt.show()