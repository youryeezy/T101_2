# T101_2

### lab-2. Линейная регрессия, полиномиальная регрессия, метод градиентного спуска

Рекомендуется использоваться Ubuntu 20.04 с установленными:
- python3
- numpy
- matplotlib
- sklearn

**Цель** Реализация и оптимизация метода градиентного спуска, решение задач регрессии

**Учебные задачи**
- освоить базовые операции _numpy_ (импорт/экспорт numpy array, перемножение матриц)
- решение задач регрессии с помощью _polyfit_
- освоить построение графиков в _matplotlib_

**Задачи**
1. Реализовать точное решение задачи поиска решения задачи регрессии (в матричном виде через обратную матрицу - использовать _pinv_, а не _inv_)

```python
def linear_regression_exact(filename):
    print("Ex1: your code here - exact solution usin invert matrix")
    return
```

2. Полиномиальная регрессия с помощью _numpy polyfit_

```python
def polynomial_regression_numpy(filename):
    print("Ex2: your code here")
    time_start = time()
    print("Ex2: your code here")
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
```

3. Реализовать самостоятельно метод градиентного спуска

```python
# Ex.3 gradient descent for linear regression without regularization

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) - gradient, i.e. partial derivatives of J over theta - dJ/dtheta_i (shape is 1 x N - the same as theta)
# x and y are both vectors

def gradient_descent_step(dJ, theta, alpha):
    print("your code goes here")

    return(theta_new)

# get gradient over all xy dataset - gradient descent
def get_dJ(x, y, theta):
    d_theta = theta
    print("your code goes here - calculate new theta")
    return d_theta   

# get gradient over all minibatch of size M of xy dataset - minibatch gradient descent
def get_dJ_minibatch(x, y, theta, M):
    d_theta = theta
    print("your code goes here - calculate new theta")
    return d_theta     

# get gradient over all minibatch of single sample from xy dataset - stochastic gradient descent
def get_dJ_sgd(x, y, theta):
    d_theta = theta
    print("your code goes here - calculate new theta")
    return d_theta    
```