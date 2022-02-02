import numpy as np
from tqdm import tqdm


# 4a
X = np.random.rand(10000, 4)
eps = np.random.normal(0, 1, 10000)
Y = []
for x, e in zip(X, eps):
    Y.append(x[0] - 2*x[1] + 3*x[2] - 4*x[3] + e)

# 4b
def rss_derivative_at_point(X, Y, Y_hat):
    DL_DY_hat = -2*(Y - Y_hat)
    DL_DW = DL_DY_hat * X
    return DL_DW

def GD(lr, init_weights=[1, 1, 1, 1], der_func=rss_derivative_at_point):
    curr_weights = init_weights
    epocs_num = 500
    # iterate over data 500 times
    for epoch in tqdm(range(epocs_num)):
        derivative = [0, 0, 0, 0]
        # calculate the derivative of each point in the data and average them
        for sample, true_Y in zip(X, Y):
            Y_hat = np.dot(sample, curr_weights)
            derivative = np.add(derivative, der_func(sample, true_Y, Y_hat))
        derivative = np.divide(derivative, len(Y))
        curr_weights = np.subtract(curr_weights, lr*derivative)
    return str(curr_weights)

weights = GD(0.1)
print('Final weights are: ' + str(weights))

# 4c
# Exponential decay of the step-size
def GD_lr_decay(lr0, init_weights=[1, 1, 1, 1], der_func=rss_derivative_at_point):
    curr_weights = init_weights
    epocs_num = 500
    # iterate over data 500 times
    for epoch in tqdm(range(epocs_num)):
        lr = lr0 * np.exp(-0.01*(epoch+1))
        derivative = [0, 0, 0, 0]
        # calculate the derivative of each point in the data and average them
        for sample, true_Y in zip(X, Y):
            Y_hat = np.dot(sample, curr_weights)
            derivative = np.add(derivative, der_func(sample, true_Y, Y_hat))
        derivative = np.divide(derivative, len(Y))
        curr_weights = np.subtract(curr_weights, lr*derivative)
    return str(curr_weights)

# weights = GD_lr_decay(0.1)
# print('Final weights are: ' + str(weights))

# SGD
def SGD(lr, init_weights=[1, 1, 1, 1], der_func=rss_derivative_at_point):
    curr_weights = init_weights
    epocs_num = 10
    batch_size = 64
    # iterate over data 10 times
    for epoch in tqdm(range(epocs_num)):
        derivative = [0, 0, 0, 0]
        # calculate the derivative of each point in the data and average them
        i = 0
        for sample, true_Y in zip(X, Y):
            Y_hat = np.dot(sample, curr_weights)
            derivative = np.add(derivative, der_func(sample, true_Y, Y_hat))
            i += 1
            if i % batch_size == 0 or i == len(Y):
                derivative = np.divide(derivative, len(Y)%batch_size if i == len(Y) else batch_size)
                curr_weights = np.subtract(curr_weights, lr*derivative)
                derivative = [0, 0, 0, 0]
    return str(curr_weights)

# weights = SGD(0.1)
# print('Final weights are: ' + str(weights))

# momentum
def GD_with_mometum(lr, init_weights=[1, 1, 1, 1], der_func=rss_derivative_at_point):
    curr_weights = init_weights
    epocs_num = 500
    gamma = 0.8
    momentum = [0, 0, 0, 0]
    # iterate over data 500 times
    for epoch in tqdm(range(epocs_num)):
        derivative = [0, 0, 0, 0]
        # calculate the derivative of each point in the data and average them
        for sample, true_Y in zip(X, Y):
            Y_hat = np.dot(sample, curr_weights)
            derivative = np.add(derivative, der_func(sample, true_Y, Y_hat))
        derivative = np.divide(derivative, len(Y))
        momentum = np.add(np.dot(momentum, gamma), np.dot(derivative, lr))
        curr_weights = np.subtract(curr_weights, momentum)
    return str(curr_weights)

##weights = GD_with_mometum(0.1)
##print('Final weights are: ' + str(weights))