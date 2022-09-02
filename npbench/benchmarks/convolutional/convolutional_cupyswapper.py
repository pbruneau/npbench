from __future__ import print_function, division
from sklearn import datasets
import math
import numpy as np
import cupy as cp
import copy
from terminaltables import AsciiTable
import progressbar
bar_widgets = ['Training: ', progressbar.Percentage(), ' ', progressbar.Bar
    (marker='-', left='[', right=']'), ' ', progressbar.ETA()]


def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in cp.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


class NeuralNetwork:
    """Neural Network. Deep Learning base model.

    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels (X, y)
    """

    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {'training': [], 'validation': []}
        self.loss_function = loss()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {'X': X, 'y': y}

    def set_trainable(self, trainable):
        """ Method which enables freezing of the weights of the network's layers. """
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """ Method which adds a layer to the neural network """
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
        self.layers.append(layer)

    def test_on_batch(self, X, y):
        """ Evaluates the model over a single batch of samples """
        y_pred = self._forward_pass(X, training=False)
        loss = cp.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        return loss, acc

    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(X)
        loss = cp.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        loss_grad = self.loss_function.gradient(y, y_pred)
        self._backward_pass(loss_grad=loss_grad)
        return loss, acc

    def fit(self, X, y, n_epochs, batch_size):
        """ Trains the model for a fixed number of epochs """
        for _ in range(n_epochs):
            batch_error = []
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size
                ):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
            self.errors['training'].append(cp.mean(batch_error))
            if self.val_set is not None:
                val_loss, _ = self.test_on_batch(self.val_set['X'], self.
                    val_set['y'])
                self.errors['validation'].append(val_loss)
        return self.errors['training'], self.errors['validation']

    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)
        return layer_output

    def _backward_pass(self, loss_grad):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def summary(self, name='Model Summary'):
        print(AsciiTable([[name]]).table)
        print('Input Shape: %s' % str(self.layers[0].input_shape))
        table_data = [['Layer Type', 'Parameters', 'Output Shape']]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params
        print(AsciiTable(table_data).table)
        print('Total Parameters: %d\n' % tot_params)

    def predict(self, X):
        """ Use the trained model to predict labels of X """
        return self._forward_pass(X, training=False)


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = cp.atleast_1d(cp.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / cp.expand_dims(l2, axis)


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
    return X_train, X_test, y_train, y_test


def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        cp.asarray(np.random.seed(cp.asnumpy(seed)))
    idx = cp.arange(X.shape[0])
    cp.random.shuffle(idx)
    return X[idx], y[idx]


def get_random_subsets(X, y, n_subsets, replacements=True):
    """ Return random subsets (with replacements) of the data """
    n_samples = cp.shape(X)[0]
    X_y = cp.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    cp.random.shuffle(X_y)
    subsets = []
    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples
    for _ in range(n_subsets):
        idx = cp.random.choice(range(n_samples), size=cp.shape(range(
            subsample_size)), replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, (-1)]
        subsets.append([X, y])
    return subsets


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = cp.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


class Adam:

    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-08
        self.m = None
        self.v = None
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_wrt_w):
        if self.m is None:
            self.m = cp.zeros(cp.shape(grad_wrt_w))
            self.v = cp.zeros(cp.shape(grad_wrt_w))
        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * cp.power(grad_wrt_w, 2)
        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)
        self.w_updt = self.learning_rate * m_hat / (cp.sqrt(v_hat) + self.eps)
        return w - self.w_updt


class Loss(object):

    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class CrossEntropy(Loss):

    def __init__(self):
        pass

    def loss(self, y, p):
        p = cp.clip(p, 1e-15, 1 - 1e-15)
        return -y * cp.log(p) - (1 - y) * cp.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(cp.argmax(y, axis=1), cp.argmax(p, axis=1))

    def gradient(self, y, p):
        p = cp.clip(p, 1e-15, 1 - 1e-15)
        return -(y / p) + (1 - y) / (1 - p)


class Sigmoid:

    def __call__(self, x):
        return 1 / (1 + cp.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax:

    def __call__(self, x):
        e_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return e_x / cp.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class TanH:

    def __call__(self, x):
        return 2 / (1 + cp.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - cp.power(self.__call__(x), 2)


class ReLU:

    def __call__(self, x):
        return cp.where(x >= 0, x, 0)

    def gradient(self, x):
        return cp.where(x >= 0, 1, 0)


class LeakyReLU:

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return cp.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return cp.where(x >= 0, 1, self.alpha)


class ELU:

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x):
        return cp.where(x >= 0.0, x, self.alpha * (cp.exp(x) - 1))

    def gradient(self, x):
        return cp.where(x >= 0.0, 1, self.__call__(x) + self.alpha)


class SELU:

    def __init__(self):
        self.alpha = 1.6732632423543772
        self.scale = 1.0507009873554805

    def __call__(self, x):
        return self.scale * cp.where(x >= 0.0, x, self.alpha * (cp.exp(x) - 1))

    def gradient(self, x):
        return self.scale * cp.where(x >= 0.0, 1, self.alpha * cp.exp(x))


class SoftPlus:

    def __call__(self, x):
        return cp.log(1 + cp.exp(x))

    def gradient(self, x):
        return 1 / (1 + cp.exp(-x))


class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return 0

    def forward_pass(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()


class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """

    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = cp.random.uniform(-limit, limit, (self.input_shape[0],
            self.n_units))
        self.w0 = cp.zeros((1, self.n_units))
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return cp.prod(self.W.shape) + cp.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = cp.sum(accum_grad, axis=0, keepdims=True)
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return self.n_units,


class RNN(Layer):
    """A Vanilla Fully-Connected Recurrent Neural Network layer.

    Parameters:
    -----------
    n_units: int
        The number of hidden states in the layer.
    activation: string
        The name of the activation function which will be applied to the output of each state.
    bptt_trunc: int
        Decides how many time steps the gradient should be propagated backwards through states
        given the loss gradient for time step t.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.

    Reference:
    http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
    """

    def __init__(self, n_units, activation='tanh', bptt_trunc=5,
        input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = activation_functions[activation]()
        self.trainable = True
        self.bptt_trunc = bptt_trunc
        self.W = None
        self.V = None
        self.U = None

    def initialize(self, optimizer):
        timesteps, input_dim = self.input_shape
        limit = 1 / math.sqrt(input_dim)
        self.U = cp.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / math.sqrt(self.n_units)
        self.V = cp.random.uniform(-limit, limit, (input_dim, self.n_units))
        self.W = cp.random.uniform(-limit, limit, (self.n_units, self.n_units))
        self.U_opt = copy.copy(optimizer)
        self.V_opt = copy.copy(optimizer)
        self.W_opt = copy.copy(optimizer)

    def parameters(self):
        return cp.prod(self.W.shape) + cp.prod(self.U.shape) + cp.prod(self
            .V.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, timesteps, input_dim = X.shape
        self.state_input = cp.zeros((batch_size, timesteps, self.n_units))
        self.states = cp.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = cp.zeros((batch_size, timesteps, input_dim))
        self.states[:, (-1)] = cp.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            self.state_input[:, (t)] = X[:, (t)].dot(self.U.T) + self.states[:,
                (t - 1)].dot(self.W.T)
            self.states[:, (t)] = self.activation(self.state_input[:, (t)])
            self.outputs[:, (t)] = self.states[:, (t)].dot(self.V.T)
        return self.outputs

    def backward_pass(self, accum_grad):
        _, timesteps, _ = accum_grad.shape
        grad_U = cp.zeros_like(self.U)
        grad_V = cp.zeros_like(self.V)
        grad_W = cp.zeros_like(self.W)
        accum_grad_next = cp.zeros_like(accum_grad)
        for t in reversed(range(timesteps)):
            grad_V += accum_grad[:, (t)].T.dot(self.states[:, (t)])
            grad_wrt_state = accum_grad[:, (t)].dot(self.V
                ) * self.activation.gradient(self.state_input[:, (t)])
            accum_grad_next[:, (t)] = grad_wrt_state.dot(self.U)
            for t_ in reversed(cp.arange(max(0, t - self.bptt_trunc), t + 1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, (t_)])
                grad_W += grad_wrt_state.T.dot(self.states[:, (t_ - 1)])
                grad_wrt_state = grad_wrt_state.dot(self.W
                    ) * self.activation.gradient(self.state_input[:, (t_ - 1)])
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)
        return accum_grad_next

    def output_shape(self):
        return self.input_shape


class Conv2D(Layer):
    """A 2D Convolution Layer.

    Parameters:
    -----------
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape: tuple
        A tuple (filter_height, filter_width).
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
    padding: string
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
    stride: int
        The stride length of the filters during the convolution over the input.
    """

    def __init__(self, n_filters, filter_shape, input_shape=None, padding=
        'same', stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True

    def initialize(self, optimizer):
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / math.sqrt(cp.prod(self.filter_shape))
        self.W = cp.random.uniform(-limit, limit, size=(self.n_filters,
            channels, filter_height, filter_width))
        self.w0 = cp.zeros((self.n_filters, 1))
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return cp.prod(self.W.shape) + cp.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        self.X_col = image_to_column(X, self.filter_shape, stride=self.
            stride, output_shape=self.padding)
        self.W_col = self.W.reshape((self.n_filters, -1))
        output = self.W_col.dot(self.X_col) + self.w0
        output = output.reshape(self.output_shape() + (batch_size,))
        return output.transpose(3, 0, 1, 2)

    def backward_pass(self, accum_grad):
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.
            n_filters, -1)
        if self.trainable:
            grad_w = accum_grad.dot(self.X_col.T).reshape(self.W.shape)
            grad_w0 = cp.sum(accum_grad, axis=1, keepdims=True)
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        accum_grad = self.W_col.T.dot(accum_grad)
        accum_grad = column_to_image(accum_grad, self.layer_input.shape,
            self.filter_shape, stride=self.stride, output_shape=self.padding)
        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=
            self.padding)
        output_height = (height + cp.sum(pad_h) - self.filter_shape[0]
            ) / self.stride + 1
        output_width = (width + cp.sum(pad_w) - self.filter_shape[1]
            ) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)


class BatchNormalization(Layer):
    """Batch normalization.
    """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.trainable = True
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None

    def initialize(self, optimizer):
        self.gamma = cp.ones(self.input_shape)
        self.beta = cp.zeros(self.input_shape)
        self.gamma_opt = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

    def parameters(self):
        return cp.prod(self.gamma.shape) + cp.prod(self.beta.shape)

    def forward_pass(self, X, training=True):
        if self.running_mean is None:
            self.running_mean = cp.mean(X, axis=0)
            self.running_var = cp.var(X, axis=0)
        if training and self.trainable:
            mean = cp.mean(X, axis=0)
            var = cp.var(X, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 -
                self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self
                .momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        self.X_centered = X - mean
        self.stddev_inv = 1 / cp.sqrt(var + self.eps)
        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta
        return output

    def backward_pass(self, accum_grad):
        gamma = self.gamma
        if self.trainable:
            X_norm = self.X_centered * self.stddev_inv
            grad_gamma = cp.sum(accum_grad * X_norm, axis=0)
            grad_beta = cp.sum(accum_grad, axis=0)
            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)
        batch_size = accum_grad.shape[0]
        accum_grad = 1 / batch_size * gamma * self.stddev_inv * (batch_size *
            accum_grad - cp.sum(accum_grad, axis=0) - self.X_centered * 
            self.stddev_inv ** 2 * cp.sum(accum_grad * self.X_centered, axis=0)
            )
        return accum_grad

    def output_shape(self):
        return self.input_shape


class PoolingLayer(Layer):
    """A parent class of MaxPooling2D and AveragePooling2D
    """

    def __init__(self, pool_shape=(2, 2), stride=1, padding=0):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True

    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, channels, height, width = X.shape
        _, out_height, out_width = self.output_shape()
        X = X.reshape(batch_size * channels, 1, height, width)
        X_col = image_to_column(X, self.pool_shape, self.stride, self.padding)
        output = self._pool_forward(X_col)
        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)
        return output

    def backward_pass(self, accum_grad):
        batch_size, _, _, _ = accum_grad.shape
        channels, height, width = self.input_shape
        accum_grad = accum_grad.transpose(2, 3, 0, 1).ravel()
        accum_grad_col = self._pool_backward(accum_grad)
        accum_grad = column_to_image(accum_grad_col, (batch_size * channels,
            1, height, width), self.pool_shape, self.stride, 0)
        accum_grad = accum_grad.reshape((batch_size,) + self.input_shape)
        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_shape[0]) / self.stride + 1
        out_width = (width - self.pool_shape[1]) / self.stride + 1
        assert out_height % 1 == 0
        assert out_width % 1 == 0
        return channels, int(out_height), int(out_width)


class MaxPooling2D(PoolingLayer):

    def _pool_forward(self, X_col):
        arg_max = cp.argmax(X_col, axis=0).flatten()
        output = X_col[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output

    def _pool_backward(self, accum_grad):
        accum_grad_col = cp.zeros((cp.prod(self.pool_shape), accum_grad.size))
        arg_max = self.cache
        accum_grad_col[arg_max, range(accum_grad.size)] = accum_grad
        return accum_grad_col


class AveragePooling2D(PoolingLayer):

    def _pool_forward(self, X_col):
        output = cp.mean(X_col, axis=0)
        return output

    def _pool_backward(self, accum_grad):
        accum_grad_col = cp.zeros((cp.prod(self.pool_shape), accum_grad.size))
        accum_grad_col[:, (range(accum_grad.size))
            ] = 1.0 / accum_grad_col.shape[0] * accum_grad
        return accum_grad_col


class ConstantPadding2D(Layer):
    """Adds rows and columns of constant values to the input.
    Expects the input to be of shape (batch_size, channels, height, width)

    Parameters:
    -----------
    padding: tuple
        The amount of padding along the height and width dimension of the input.
        If (pad_h, pad_w) the same symmetric padding is applied along height and width dimension.
        If ((pad_h0, pad_h1), (pad_w0, pad_w1)) the specified padding is added to beginning and end of
        the height and width dimension.
    padding_value: int or tuple
        The value the is added as padding.
    """

    def __init__(self, padding, padding_value=0):
        self.padding = padding
        self.trainable = True
        if not isinstance(padding[0], tuple):
            self.padding = (padding[0], padding[0]), padding[1]
        if not isinstance(padding[1], tuple):
            self.padding = self.padding[0], (padding[1], padding[1])
        self.padding_value = padding_value

    def forward_pass(self, X, training=True):
        output = cp.pad(X, pad_width=((0, 0), (0, 0), self.padding[0], self
            .padding[1]), mode='constant', constant_values=self.padding_value)
        return output

    def backward_pass(self, accum_grad):
        pad_top, pad_left = self.padding[0][0], self.padding[1][0]
        height, width = self.input_shape[1], self.input_shape[2]
        accum_grad = accum_grad[:, :, pad_top:pad_top + height, pad_left:
            pad_left + width]
        return accum_grad

    def output_shape(self):
        new_height = self.input_shape[1] + cp.sum(self.padding[0])
        new_width = self.input_shape[2] + cp.sum(self.padding[1])
        return self.input_shape[0], new_height, new_width


class ZeroPadding2D(ConstantPadding2D):
    """Adds rows and columns of zero values to the input.
    Expects the input to be of shape (batch_size, channels, height, width)

    Parameters:
    -----------
    padding: tuple
        The amount of padding along the height and width dimension of the input.
        If (pad_h, pad_w) the same symmetric padding is applied along height and width dimension.
        If ((pad_h0, pad_h1), (pad_w0, pad_w1)) the specified padding is added to beginning and end of
        the height and width dimension.
    """

    def __init__(self, padding):
        self.padding = padding
        if isinstance(padding[0], int):
            self.padding = (padding[0], padding[0]), padding[1]
        if isinstance(padding[1], int):
            self.padding = self.padding[0], (padding[1], padding[1])
        self.padding_value = 0


class Flatten(Layer):
    """ Turns a multidimensional matrix into two-dimensional """

    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return cp.prod(self.input_shape),


class UpSampling2D(Layer):
    """ Nearest neighbor up sampling of the input. Repeats the rows and
    columns of the data by size[0] and size[1] respectively.

    Parameters:
    -----------
    size: tuple
        (size_y, size_x) - The number of times each axis will be repeated.
    """

    def __init__(self, size=(2, 2), input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.size = size
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        X_new = X.repeat(self.size[0], axis=2).repeat(self.size[1], axis=3)
        return X_new

    def backward_pass(self, accum_grad):
        accum_grad = accum_grad[:, :, ::self.size[0], ::self.size[1]]
        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        return channels, self.size[0] * height, self.size[1] * width


class Reshape(Layer):
    """ Reshapes the input tensor into specified shape

    Parameters:
    -----------
    shape: tuple
        The shape which the input shall be reshaped to.
    """

    def __init__(self, shape, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.shape = shape
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0],) + self.shape)

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return self.shape


class Dropout(Layer):
    """A layer that randomly sets a fraction p of the output units of the previous layer
    to zero.

    Parameters:
    -----------
    p: float
        The probability that unit x is set to zero.
    """

    def __init__(self, p=0.2):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, X, training=True):
        c = 1 - self.p
        if training:
            self._mask = cp.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward_pass(self, accum_grad):
        return accum_grad * self._mask

    def output_shape(self):
        return self.input_shape


activation_functions = {'relu': ReLU, 'sigmoid': Sigmoid, 'selu': SELU,
    'elu': ELU, 'softmax': Softmax, 'leaky_relu': LeakyReLU, 'tanh': TanH,
    'softplus': SoftPlus}


class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return 'Activation (%s)' % self.activation_func.__class__.__name__

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


def determine_padding(filter_shape, output_shape='same'):
    if output_shape == 'valid':
        return (0, 0), (0, 0)
    elif output_shape == 'same':
        filter_height, filter_width = filter_shape
        pad_h1 = int(math.floor((filter_height - 1) / 2))
        pad_h2 = int(math.ceil((filter_height - 1) / 2))
        pad_w1 = int(math.floor((filter_width - 1) / 2))
        pad_w2 = int(math.ceil((filter_width - 1) / 2))
        return (pad_h1, pad_h2), (pad_w1, pad_w2)


def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + cp.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + cp.sum(pad_w) - filter_width) / stride + 1)
    i0 = cp.repeat(cp.arange(filter_height), filter_width)
    i0 = cp.tile(i0, channels)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(filter_width), filter_height * channels)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = cp.repeat(cp.arange(channels), filter_height * filter_width).reshape(
        -1, 1)
    return k, i, j


def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    images_padded = cp.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode=
        'constant')
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w),
        stride)
    cols = images_padded[:, (k), (i), (j)]
    channels = images.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width *
        channels, -1)
    return cols


def column_to_image(cols, images_shape, filter_shape, stride, output_shape=
    'same'):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + cp.sum(pad_h)
    width_padded = width + cp.sum(pad_w)
    images_padded = cp.zeros((batch_size, channels, height_padded,
        width_padded))
    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w),
        stride)
    cols = cols.reshape(channels * cp.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    cp.asarray(np.add.at(cp.asnumpy(images_padded), cp.asnumpy((slice(None),
        k, i, j)), cp.asnumpy(cols)))
    return images_padded[:, :, pad_h[0]:height + pad_h[0], pad_w[0]:width +
        pad_w[0]]


def main(X_train, X_test, y_train, y_test, f):
    optimizer = Adam()
    clf = NeuralNetwork(optimizer=optimizer, loss=CrossEntropy,
        validation_data=(X_test, y_test))
    clf.add(Conv2D(n_filters=f, filter_shape=(3, 3), stride=1, input_shape=
        (1, 8, 8), padding='same'))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Conv2D(n_filters=f * 2, filter_shape=(3, 3), stride=1, padding=
        'same'))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Flatten())
    clf.add(Dense(256))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.4))
    clf.add(BatchNormalization())
    clf.add(Dense(10))
    clf.add(Activation('softmax'))
    _ = clf.fit(X_train, y_train, n_epochs=50, batch_size=256)
    _, accuracy = clf.test_on_batch(X_test, y_test)
