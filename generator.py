import numpy
import numpy as np
import matplotlib.pyplot as plt
from time import time


def generate_linear(a, b, noise, filename, size=100):  # generate x,y for linear
    x = 2 * np.random.rand(size, 1) - 1
    y = a * x + b + noise * a * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')
    return x, y


def linear_regression_numpy(filename):  # linear regression with polyfit
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    print("shape of x and y are: ", np.shape(x), np.shape(y))

    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
    h = model[0] * x + model[1]

    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return model


def linear_regression_exact(filename):  # custom linear regression
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    time_start = time()
    tmp_x = np.hstack([np.ones((100, 1)), x])
    trans_x = np.transpose(tmp_x)
    res_theta = np.linalg.matrix_power(trans_x.dot(tmp_x), -1).dot(trans_x).dot(y)
    print("Res_theta: ", res_theta)
    print("Shape of x and y are: ", np.shape(x), np.shape(y))
    time_end = time()

    h = res_theta[1] * x + res_theta[0]
    print(f"Linear regression time:{time_end - time_start}")
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return res_theta


def check(modl, ground_truth):
    if len(modl) != len(ground_truth):
        print("Model is inconsistent")
        return False
    else:
        r = np.dot(modl - ground_truth, modl - ground_truth) / (np.dot(ground_truth, ground_truth))
        print("Result of check: ", r)
        if r < 0.001:
            return True
        else:
            return False


def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    if len(a) != (n + 1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n}')
        return
    for i in range(0, n + 1):
        y = y + a[i] * np.power(x, i) + noise * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')


def polynomial_regression_numpy(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    print("Shape of x and y are: ", np.shape(x), np.shape(y))
    time_start = time()
    modl = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 2)
    time_end = time()
    print(f"Polynomial regression with polyfit in {time_end - time_start} seconds")
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    x = np.sort(x, axis=0)
    h = modl[0] * x * x + modl[1] * x + modl[2]

    plt.plot(x, h, "r", label='modl')

    plt.legend()
    plt.show()
    return modl


# Ex.2 gradient descent for linear regression without regularization

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) - gradient, i.e. partial derivatives of J over theta - dJ/dtheta_i (shape is 1 x N - the same as theta)
# x and y are both vectors

def gradient_descent_step(dJ, theta, alpha):
    return theta


# get gradient over all xy dataset - gradient descent
# get gradient over all xy dataset - gradient descent
def get_dJ(x, y, theta):
    alf = 0.001
    h = theta[1] * x + theta[0]
    theta_new = theta
    for tet in theta_new:
        tet = tet - alf * np.sum((h - y) * x)  # частная производная

    print('theta_new = ', theta_new)
    return theta_new


# get gradient over all minibatch of size M of xy dataset - minibatch gradient descent
def get_dJ_minibatch(x, y, theta, M):
    theta_new = theta
    print("your code goes here - calculate new theta")
    return theta_new


# get gradient over all minibatch of single sample from xy dataset - stochastic gradient descent
def get_dJ_sgd(x, y, theta):
    theta_new = theta
    print("your code goes here - calculate new theta")
    return theta_new


# try each of gradient decent (complete, minibatch, sgd) for varius alphas
# L - number of iterations
# plot results as J(i)
def minimize(theta, x, y, L):
    n = 100
    theta = np.zeros(n)  # you can try random initialization
    dJ = np.zeros(n)
    for i in range(0, L):
        theta = get_dJ(x, y, theta)  # here you should try different gradient descents
        J = 0  # here you should calculate it properly

    plt.title("Minimize task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    generate_linear(1, -3, 1, 'linear.csv', 100)
    model = np.squeeze(linear_regression_exact("linear.csv"))
    mod1 = np.squeeze(numpy.asarray(np.array(([-3], [1]))))
    print(f"Is model correct? - {check(model, mod1)}")
    print("*" * 40)

    generate_poly([1, 2, 3], 2, 0.5, 'polynomial.csv')
    poly_model = polynomial_regression_numpy("polynomial.csv")

    # ex2. find minimum with gradient descent
    # 0. generate date with function above
    # 1. shuffle data into train - test - valid
    # 2. call minuimize(...) and plot J(i)
    # 3. call check(theta1, theta2) to check results for optimal theta

    # ex3. polinomial regression
    # 0. generate date with function generate_poly for degree=3, use size = 10, 20, 30, ... 100
    # for each size:
    # 1. shuffle data into train - test - valid
    # Now we're going to try different degrees of model to aproximate our data, set degree=1 (linear regression)
    # 2. call minimize(...) and plot J(i)
    # 3. call check(theta1, theta2) to check results for optimal theta
    # 4. plot min(J_train), min(J_test) vs size: is it overfit or underfit?
    #
    # repeat 0-4 for degres = 2,3,4

    # ex3* the same with regularization
