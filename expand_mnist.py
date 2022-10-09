import pickle
from numpy import ndarray, delete, row_stack, column_stack, zeros, reshape

# The performance of artificial neural networks depends on the data, the algorithm the is trained on,
# as much as the architecture of the network. Hence, performance can often be enhanced by working with
# more data. When no additional data is available, a good way to expose the network to more data is by
# artificially expanding the training data. The subsequent algorithm performs this task by shifting the
# images of the digits to the left, right, up and down. Normally, a seven remains a seven, even if the
# pixels are shifted up. For the ANN, the two images -- the original and the derivation -- are completely
# different however, as they will become completely different input tensors.


def expand_data():

    print("Expanding the MNIST training dataset")

    # get data
    f = open("saved_files/data.pickle", 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    expanded_training_pairs = []

    # loop through existing training samples
    # shift data points slightly up, left, down, right
    for sample in training_data:
        x = sample[0]
        y = sample[1]
        x_0: ndarray = reshape(x, (28, 28))

        # cut sample
        x_1_cut: ndarray = delete(x_0, 0, 0)  # up
        x_2_cut: ndarray = delete(x_0, 0, 1)  # left
        x_3_cut: ndarray = delete(x_0, -1, 0)  # down
        x_4_cut: ndarray = delete(x_0, -1, 1)  # right

        # add zeros
        x_1: ndarray = row_stack((x_1_cut, zeros(28,))).reshape((784,))
        x_2: ndarray = column_stack((x_2_cut, zeros(28,))).reshape((784,))
        x_3: ndarray = row_stack((zeros(28,), x_3_cut)).reshape((784,))
        x_4: ndarray = column_stack((zeros(28,), x_4_cut)).reshape((784,))

        # append to training data
        x_pair: tuple = (x, y)
        x_1_pair: tuple = (x_1, y)
        x_2_pair: tuple = (x_2, y)
        x_3_pair: tuple = (x_3, y)
        x_4_pair: tuple = (x_4, y)
        expanded_training_pairs.append(x_pair)
        expanded_training_pairs.append(x_1_pair)
        expanded_training_pairs.append(x_2_pair)
        expanded_training_pairs.append(x_3_pair)
        expanded_training_pairs.append(x_4_pair)

    # save data
    file = open('saved_files/data.pickle', 'wb')
    pickle.dump((expanded_training_pairs, validation_data, test_data), file)
    print('Executed data expansion')
