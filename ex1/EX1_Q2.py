import numpy as np


def init(layers):
    np.random.seed(42)

    params_w = {}

    for index in range(len(layers)-1):

        layer_num = index + 1
        in_layer_size = layers[index]
        out_layer_size = layers[index + 1]

        params_w['weight' + str(layer_num)] = np.ones((out_layer_size, in_layer_size))
    return params_w

def sigmoid(input):
    return 1/(1 + np.exp(-input))

#relu activation
def relu(input):
    return np.maximum(input, 0)

def identity(input):
    return input

#derivate of a sigmoid w.r.t. input
def d_sigmoid(d_init, out):
    sig = sigmoid(out)
    return d_init * sig * (1 - sig)

#derivate of a relu w.r.t. input
def d_relu(d_init, out):
    d = np.array(d_init, copy = True)
    d[out < 0] = 0.
    return d

def d_identitiy(d_init, out):
    return 1

def one_layer_forward_pass(input_activations, weights, activation='R'):
    output = np.dot(weights, input_activations)

    if activation is 'R':
        activation_next = relu(output)
    else:
        activation_next = sigmoid(output)

    return activation_next, output

def forward_pass(X, params_w, layers, activate):

    num_layers = len(layers) - 1

    activation_dict = {}
    output_dict = {}

    curr_act = X

    for index in range(num_layers):

        layer_index = index + 1
        prev_act = curr_act

        curr_weight = params_w["weight" + str(layer_index)]

        curr_act, curr_out = one_layer_forward_pass(prev_act, curr_weight, activate[index])

        activation_dict["act" + str(index)] = prev_act
        output_dict["out" + str(layer_index)] = curr_out

    return curr_act, activation_dict, output_dict

def one_layer_backward_pass(curr_grad, curr_weight, curr_out, prev_act, activation='R'):
    # how many sample in previous activations?
    num = prev_act.shape[1]

    # find out what we are differentiating
    if activation is 'R':
        d_act_func = d_relu
    elif activation is 'S':
        d_act_func = d_sigmoid

    # derivative of activation function
    d_curr_out = d_act_func(curr_grad, curr_out)

    # derivative of weight matrix
    d_curr_weight = np.dot(d_curr_out, prev_act.T) / num

    # derivative of input activations from previous layer
    d_prev_act = np.dot(curr_weight.T, d_curr_out)

    return d_prev_act, d_curr_weight

def backward_pass(y_pred, train_Y, activation_dict, output_dict, params_w, layers, activate):
    gradients = {}

    num_samples = train_Y.shape[0]

    train_Y = train_Y.reshape(y_pred.shape)

    # derivative of RSS function w.r.t. predictions
    d_prev_act = (-2*np.subtract(train_Y, y_pred)).reshape(1, num_samples)

    num_layers = len(layers) - 1
    layer_num = [x + 1 for x in range(num_layers)]
    layer_num.reverse()

    activate_ = activate
    activate_.reverse()

    for index, layer_num in enumerate(layer_num):
        activation = activate_[layer_num - 1]

        d_curr_act = d_prev_act

        prev_act = activation_dict['act' + str(layer_num - 1)]  # activations are one index behind
        curr_out = output_dict['out' + str(layer_num)]

        curr_weight = params_w['weight' + str(layer_num)]

        d_prev_act, d_curr_weight = one_layer_backward_pass(d_curr_act, curr_weight, curr_out,
                                                                         prev_act, activation)

        gradients["d_weight" + str(layer_num)] = d_curr_weight

    return gradients

X = np.array([1, 2, -1]).reshape(3, 1)
Y = np.array([0])
layers = [3, 2, 2, 1]
params_w = init(layers)
curr_act, activation_dict, output_dict = forward_pass(X, params_w, layers, activate=['R', 'R', 'R'])
gradients = backward_pass(curr_act[0], Y, activation_dict, output_dict, params_w, layers, activate=['R', 'R', 'R'])
print(gradients)

