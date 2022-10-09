import pickle
import gzip
import numpy as np

# This code loads the mnist dataset. It is based on python 2 code presented in Neural Networks
# and Deep Learning by Micheal A. Nielsen. Slight modifications were made such that the program
# runs on python 3.


def load_data() -> tuple:
    # change file
    file = gzip.open('/Users/benjaminschoofs/Documents/GitHub/neural-networks-and-deep-learning-master/data/mnist'
                     '.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(file, encoding='latin1')
    file.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784,)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784,)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784,)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    with open('saved_files/data.pickle', 'wb') as file:
        pickle.dump((training_data, validation_data, test_data), file)


def fetch_data() -> tuple:
    (training_data, validation_data, test_data) = pickle.load(open('saved_files/data.pickle', 'rb'))
    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10,))
    e[j] = 1.0
    return e
