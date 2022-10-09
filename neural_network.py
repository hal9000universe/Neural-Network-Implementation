from numpy import array, random, ndarray, transpose, \
    vectorize, float64, exp, ones, argmax, row_stack, \
    sqrt, log, linalg
from pickle import dump, load
from numba import njit, jit
from mnist_loader import fetch_data, load_data_wrapper
from expand_mnist import expand_data


# This code is a fully matrix-based implementation of a feedforward, fully-connected neural network.
# It solves the MNIST classification problem with an accuracy of 90.8%.


@njit
def sigmoid(z) -> float:
    return 1.0 / (1.0 + exp(-z))


@njit
def dif_sigmoid(z) -> float:
    return sigmoid(z) * (1.0 - sigmoid(z))


@njit
def relu(z) -> float:
    z = z / 10.0
    return max(0.0, z)


@njit
def dif_relu(z) -> float:
    if z >= 0.0:
        return 1.0
    else:
        return 0.0


@njit
def identity(z) -> float:
    return z


@njit
def dif_identity(z: ndarray):
    return ones(z.shape, dtype=float64)


class Neuron:

    def __call__(self, z):
        raise NotImplementedError

    @staticmethod
    def activation(z):
        raise NotImplementedError

    @staticmethod
    def dif_activation(z):
        raise NotImplementedError


class Sigmoid(Neuron):

    def __call__(self, z):
        return self.activation(z)

    @staticmethod
    def activation(z):
        return vectorize(sigmoid)(z)

    @staticmethod
    def dif_activation(z):
        return vectorize(dif_sigmoid)(z)


class ReLU(Neuron):

    def __call__(self, z):
        return self.activation(z)

    @staticmethod
    def activation(z):
        return vectorize(relu)(z)

    @staticmethod
    def dif_activation(z):
        return vectorize(dif_relu)(z)


class Identity(Neuron):

    def __call__(self, z):
        return z

    @staticmethod
    def activation(z):
        return vectorize(identity)(z)

    @staticmethod
    def dif_activation(z):
        return ones(z.shape)


def extend_vector(vector: ndarray, dim: int) -> ndarray:
    # extend bias vector to bias matrix to handle entire batches simultaneously
    vectors = array([vector for _ in range(dim)])
    vec_matrix: ndarray = row_stack(vectors)
    return vec_matrix


def collapse(m: ndarray, num_ones: int):
    # collapse bias matrix back to bias vector to process single inputs
    return transpose(m) @ ones(num_ones, dtype=float64) / num_ones


class Layer:

    def __init__(self, input_dim: int, output_dim: int, activation: Neuron):
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.activation: Neuron = activation
        # weights initialized with a gaussian distribution with standard deviation sqrt(input_dim)
        self.weight_matrix: ndarray = random.rand(input_dim, output_dim) / sqrt(self.input_dim)
        self.bias: ndarray = random.rand(output_dim)

    def __call__(self, x: ndarray) -> ndarray:
        x: ndarray = x @ self.weight_matrix
        x: ndarray = x + self.bias
        return x

    def activation(self, z):
        return self.activation(z)


class InputLayer(Layer):

    def __call__(self, x: ndarray):
        return x

    def activation(self, z):
        return z


class Loss:

    @staticmethod
    def loss(a, y):
        raise NotImplementedError

    @staticmethod
    def error(a, y, dif_ac_z):
        raise NotImplementedError


class MSE(Loss):
    """
    This loss function is meant to be used with piece-wise linear functions.
    """

    @staticmethod
    def loss(a, y):
        return linalg.norm(a - y) ** 2

    @staticmethod
    def error(a, y, dif_ac_z):
        return (a - y) * dif_ac_z


def cross_entropy(a, y):
    return -(y * log(a) + (1 - y) * log(1 - a))


class CrossEntropy(Loss):
    """
    This loss function is meant to be combined with sigmoid neurons.
    """

    @staticmethod
    def loss(a, y):
        return sum(sum(vectorize(cross_entropy)(a, y))) / (batch_size * 10)

    @staticmethod
    def error(a, y, dif_ac_z):
        return a - y


class Network:

    def __init__(self):
        self.layers: list = []
        self.train_mode: bool = False
        self.loss: Loss = CrossEntropy()

    def train(self):
        """
        Switches to training mode:
        bias vectors are extended to bias
        matrices to handle batch data
        simultaneously.
        """
        for layer in self.layers:
            layer.bias = extend_vector(layer.bias, batch_size)
        self.train_mode = True

    def eval(self):
        """
        Switches to evaluation mode:
        bias matrices are collapsed to
        bias vectors to handle single
        test and validation samples.
        """
        for layer in self.layers:
            # compute biases averaged over training samples
            layer.bias = collapse(layer.bias, batch_size)
        self.train_mode = False

    def forward(self, x: ndarray):
        """
        Passes input forward through the network.
        """
        z_mem = []  # store weighted sums
        activations = []  # store activations
        ac = x
        for layer in range(len(self.layers)):
            z: ndarray = self.layers[layer](ac)
            ac: ndarray = self.layers[layer].activation(z)
            if self.train_mode:
                z_mem.append(z)
                activations.append(ac)
        if self.train_mode:
            return ac, z_mem, activations
        else:
            return ac

    def __call__(self, x: ndarray):
        return self.forward(x)

    def evaluate(self, test_set: list) -> float:
        correct = 0
        total = len(test_set)
        for feature, label in test_set:
            out = self.forward(feature)
            prediction = argmax(out)
            if prediction == label:
                correct += 1
        return correct / total

    @staticmethod
    def gen_batches(train_data: list, mini_batch_size: int) -> list:
        """
        generates a list of batches.

        train_data: list of training data
        mini_batch_size: size of mini-batches
        return: list of batches:

        batches[i][0]: vector representation of image
        batches[i][1]: label of image
        """
        random.shuffle(train_data)
        n = len(train_data)
        batches: list = [train_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
        batches.pop(-1)
        return batches

    def step(self, batches: list, lr: float, rc: float):
        """
        batch_data: list of training samples containing
        an input at index 0 and a label at index 1.
        lr: learning rate
        rc: regularization constant for L2 regularization
        """
        inp: ndarray = array([batches[sample][0] for sample in range(len(batches))])
        y: ndarray = array([batches[sample][1] for sample in range(len(batches))])
        # forward pass
        out, z_mem, activations = self.forward(inp)
        # backward pass
        loss: Loss = self.loss
        dif_ac_z: ndarray = self.layers[-1].activation.dif_activation(z_mem[-1])
        # compute error
        error: ndarray = loss.error(out, y, dif_ac_z)
        # change of weights averaged over training samples
        update_matrix_w = transpose(activations[-2]) @ error / batch_size
        # update biases
        self.layers[-1].bias = self.layers[-1].bias - lr * error
        # L2 regularization
        self.layers[-1].weight_matrix = (1 - lr * rc / batch_size) * self.layers[-1].weight_matrix
        # update weights
        self.layers[-1].weight_matrix = self.layers[-1].weight_matrix - lr * update_matrix_w
        # back-propagate error to update the remaining parameters
        self.backprop(error, activations, lr, rc)

    def backprop(self, error: ndarray, activations: list, lr: float, rc: float):
        """
        back-propagates the error and updates parameters.

        error: ndarray - error in the last layer for training samples in the batch
        activations: list - activations of all layers for training samples in the batch
        lr: float - learning rate
        rc: float - regularization constant
        """
        for layer in range(2, len(self.layers)):
            dif_ac_z = self.layers[-layer].activation.dif_activation(activations[-layer])
            # multi-variable chain-rule: sum over all intermediate variables
            error = (error @ transpose(self.layers[-layer + 1].weight_matrix)) * dif_ac_z
            # changes of weights averaged over training samples
            update_matrix_w = transpose(activations[-layer - 1]) @ error / batch_size
            # update biases
            self.layers[-layer].bias = self.layers[-layer].bias - lr * error
            # L2 regularization
            self.layers[-layer].weight_matrix = (1 - lr * rc / batch_size) * self.layers[-layer].weight_matrix
            # update weights
            self.layers[-layer].weight_matrix = self.layers[-layer].weight_matrix - lr * update_matrix_w

    # Stochastic Gradient Descent
    def SGD(self, train_data: list, mini_batch_size: int, num_epochs: int, lr: float, rc: float):
        """
        performs stochastic gradient descent.

        train_data: list of training data
        mini_batch_size: int stating the size of the mini-batches
        num_epochs: int indicating the number of times the algorithm will go through the training data
        lr: float - learning rate
        rc: float - regularization constant
        """
        for epoch in range(1, num_epochs):
            self.train()
            batches = self.gen_batches(train_data, mini_batch_size)
            for batch in range(len(batches)):
                # update parameters
                self.step(batches[batch], lr, rc)
            lr = lr * 0.95  # dynamic learning rate
            self.eval()
            accuracy = self.evaluate(validation_data) * 100
            print("Epoch: {} --- accuracy: {}".format(epoch, accuracy))

    @staticmethod
    def save_model(network):
        with open('saved_files/model.pickle', 'wb') as file:
            dump(network, file)

    @staticmethod
    def load_model():
        return load(open('saved_files/model.pickle', 'rb'))

    def accuracy(self, x: ndarray(shape=(2,)), args: tuple) -> float:
        train_data: list = args[1][0:10000]  # use less training data for the sake of efficiency
        random.shuffle(train_data)
        self.SGD(train_data, batch_size, args[0], x[0], x[1])
        accuracy = self.evaluate(validation_data)
        print("Accuracy: {}".format(accuracy * 100))
        return accuracy

    def reinit(self):
        print("Reinitialize ...")
        for layer in self.layers:
            layer.weight_matrix = random.rand(layer.input_dim, layer.output_dim) / sqrt(layer.input_dim)
            layer.bias = random.rand(layer.output_dim)


class ImageClassification(Network):

    def __init__(self):
        super(ImageClassification, self).__init__()

        # layer definitions
        self.input_layer: InputLayer = InputLayer(784, 784, Identity())
        self.sigmoid1: Layer = Layer(784, 64, ReLU())
        self.sigmoid2: Layer = Layer(64, 10, ReLU())

        self.layers = [self.input_layer, self.sigmoid1, self.sigmoid2]
        self.loss: Loss = MSE()


def run_model(network: Network):
    learning_rate, regularization_constant = 0.2, 0.2
    print("Training ...")
    network.SGD(training_data, batch_size, epochs, learning_rate, regularization_constant)
    print("Final accuracy: {}".format(network.evaluate(test_data) * 100))


if __name__ == '__main__':
    # load_data_wrapper()
    # expand_data()  # uncomment to expand the training data
    # (use only once since the expanded training dataset is saved automatically)
    training_data, validation_data, test_data = fetch_data()

    epochs: int = 10
    batch_size: int = 64

    model: Network = ImageClassification()

    # pre-run
    print('Initial accuracy: {}'.format(model.evaluate(test_data) * 100))

    run_model(model)

    model.save_model(model)
    model = model.load_model()
