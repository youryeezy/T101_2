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
    modl = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    print('modl - ', modl)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
    h = modl[0] * x + modl[1]

    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return modl


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
def get_dJ(x, y, theta):
    alf = 0.001

    # tmp_x = np.hstack([np.ones((100, 1)), x])
    trp_theta = np.transpose(theta)
    h = trp_theta.dot(x)
    buf = h - y
    for i in range(len(h)):
        buf[i] *= x[i]
    theta_new = theta
    buf_sum = np.sum(buf)
    for i in range(len(x)):
        theta_new[i] = theta_new[i] - alf * buf_sum  # тета - альфа * частная производная
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
def minimize(filename, L):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    print("Shape of x and y are: ", np.shape(x), np.shape(y))

    n = 2
    theta_stud = np.zeros(n)  # you can try random initialization
    dJ = np.zeros(n)

    # theta_stud = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    for i in range(0, L):
        theta_stud = get_dJ(x, y, theta_stud)  # here you should try different gradient descents

    h = theta_stud[1] * x + theta_stud[0]
    J = 1
    print(np.shape(J))
    plt.title("Minimize task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(theta_stud, J, "b.")
    plt.legend()
    plt.show()
    return theta_stud


def shuffle(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)

    x_stud = x[0:70]
    y_stud = x[0:70]
    data = np.hstack((x_stud, y_stud))
    np.savetxt('stud.csv', data, delimiter=',')

    x_test = x[70:85]
    y_test = x[70:85]
    data = np.hstack((x_test, y_test))
    np.savetxt('test.csv', data, delimiter=',')

    x_valid = x[85:100]
    y_valid = x[85:100]
    data = np.hstack((x_valid, y_valid))
    np.savetxt('valid.csv', data, delimiter=',')


if __name__ == "__main__":
    # ex1. exact solution
    generate_linear(1, -3, 1, 'linear.csv', 100)
    model = np.squeeze(linear_regression_exact("linear.csv"))
    mod1 = np.squeeze(numpy.asarray(np.array(([-3], [1]))))
    print(f"Is model correct? - {check(model, mod1)}")
    print("*" * 40)

    # ex1. polynomial with numpy
    generate_poly([1, 2, 3], 2, 0.5, 'polynomial.csv')
    poly_model = polynomial_regression_numpy("polynomial.csv")
    print("*" * 40)

    # ex2. find minimum with gradient descent
    # 0. generate date with function above
    generate_linear(1, -3, 1, 'linear.csv', 100)
    # 1. shuffle data into train - test - valid
    shuffle('linear.csv')
    # 2. call minimize(...) and plot J(i)
    theta = minimize('stud.csv', 10)
    # 3. call check(theta1, theta2) to check results for optimal theta
    check(theta[0], theta[1])
    print("*" * 40)

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
