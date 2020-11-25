import numpy as np
from minimization_methods import *
from tools import parameter_instantiate as hhg
from multiprocessing import Pool
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

"""Hubbard model Parameters"""
L = 10  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
pbc = True

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm

# target parameters
target_U = 1 * t0
target_a = 4

lat = hhg(field, N_up, N_down, L, 0, target_U, t0, F0=F0, a=target_a, pbc=pbc)
cycles = 10
n_steps = 2000
start = 0
stop = cycles / lat.freq
target_delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)[1]
# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, target_delta, pbc)
params.set_basis()

# bounds for variables
U_upper = 10 * params.t
U_lower = 0
a_upper = 10
a_lower = 0
bounds = ((U_lower, U_upper), (a_lower, a_upper))


def add_training(num, mode):
    """
    Adds training examples to data.csv. For each row of the file, the first column is the expected value of U/t0, the
    second is the expected value of a, and the remaining columns contain the values of the spectrum.
    :param num: number of randomized U and a values for calculation
    :param mode: a string to open the file in a particular mode, either "w" or "a"
    :return: None
    """

    U_vals = (U_upper - U_lower) * np.random.random(num) + U_lower
    a_vals = (a_upper - a_lower) * np.random.rand(num) + a_lower

    # U_vals, a_vals = np.meshgrid(U_vals, a_vals)
    # U_vals = list(U_vals.ravel())
    # a_vals = list(a_vals.ravel())

    x_vals = list(zip(U_vals, a_vals))

    print(len(x_vals))
    data_points = ((np.array(point), params) for point in x_vals)

    data = {}

    num_threads = 100

    # get current spectrums
    with Pool(num_threads) as pool:

        counter = 0
        for res in pool.starmap(current_expectation_power_spectrum, data_points):
            res = res / max(res)  # normalize data before saving
            data[x_vals[counter]] = res
            counter += 1
            print("Added!")

        pool.close()

    # write data to files
    with open("./TrainingData/data.csv", mode, newline='') as f:
        w = csv.writer(f)
        for x, current in data.items():
            row = [z for z in current]
            row.insert(0, x[1])
            row.insert(0, x[0])
            w.writerow(row)
            del row


def read_training_data():
    """
    Reads in training data from data.csv
    :return: a list of tuples of the form (expected x, input spectrum)
    """
    with open("./TrainingData/data.csv", "r", newline='') as f:
        reader = csv.reader(f)
        data = [(np.array([float(row[0]), float(row[1])]), np.array([float(z) for z in row[2:]])) for row in reader]

    return data


def split_data(data, test_proportion):
    """
    Splits data into training and testing data
    :param data: a list of tuples of the form (expected x, input spectrum)
    :param test_proportion: proportion of the total data that will be returned as test data
    :return: training_input, training_output, test_input, test_output which are all numpy arrays
    """

    # shuffle and split data into training and test data
    data = np.array(data)
    np.random.shuffle(data)
    split_index = int(len(data) * (1 - test_proportion))
    training_data, test_data = data[:split_index], data[split_index:]

    training_input = np.array([training_data[i][1] for i in range(len(training_data))])
    training_output = np.array([training_data[i][0] for i in range(len(training_data))])
    test_input = np.array([test_data[i][1] for i in range(len(test_data))])
    test_output = np.array([test_data[i][0] for i in range(len(test_data))])

    return training_input, training_output, test_input, test_output


def train_linear_regression(data):
    """
    This method trains the model using a linear regression
    :param data: a list of tuples of the form (expected x, input spectrum)
    :return:
    """

    training_input, training_output, test_input, test_output = split_data(data, .1)

    spectrum_length = len(data[0][1])

    linear_model = keras.Sequential()
    linear_model.add(keras.Input(shape=(spectrum_length,)))
    linear_model.add(layers.Dense(2))

    # compile model
    # linear_model.compile(optimizer=tf.optimizers.Adam(), loss=keras.losses.mean_squared_error)
    linear_model.compile(optimizer=tf.optimizers.Adam(), loss=keras.losses.mean_absolute_error)

    print("Fitting model")
    history = linear_model.fit(
        training_input, training_output,
        epochs=1000, verbose=2
    )

    print(history.history)

    print("Evaluating test data")
    linear_model.evaluate(test_input, test_output)


def train_dnn(data, num_epochs, num_hidden_layers, layer_size, loss_func, test_proportion=.1):
    """
    Trains a dnn, optimizer is set a Adam and activation func is set as rectified linear unit.
    :param data: a list of tuples of the form (expected x, input spectrum)
    :param num_epochs: number of epochs for training
    :param num_hidden_layers: number of hidden layers to add (0 is a linear regression)
    :param layer_size: number of neurons in a hidden layer
    :param loss_func: the loss function for training
    :param test_proportion: a float describing the proportion of data to be used for testing
    :return: the neural network that has been trained, training loss, test loss
    """
    training_input, training_output, test_input, test_output = split_data(data, test_proportion)

    spectrum_length = len(data[0][1])

    dnn = keras.Sequential()
    dnn.add(keras.Input(shape=(spectrum_length,)))  # input layer
    # hidden layers
    for _ in range(num_hidden_layers):
        dnn.add(layers.Dense(layer_size, activation="relu"))
    dnn.add(layers.Dense(2))  # output layer

    # compile model
    dnn.compile(optimizer=keras.optimizers.Adam(), loss=loss_func)

    print("Fitting model")
    history = dnn.fit(
        training_input, training_output,
        epochs=num_epochs, verbose=2
    )

    training_loss = history.history['loss'][-1]

    print("Evaluating test data")
    test_loss = dnn.evaluate(test_input, test_output)

    return dnn, training_loss, test_loss


# def save_dnn(dnn, json_file, h5_file):
#     model_json = dnn.to_json()
#     with open(json_file, "w") as f:
#         f.write(model_json)
#     dnn.save_weights(h5_file)
#
#
# def load_dnn(json_file, h5_file, loss_func, optimizer):
#     with open(json_file, "r") as f:
#         dnn = f.read()
#     dnn = model_from_json(dnn)
#     dnn.load_weights(h5_file)
#
#     dnn.compile(loss=loss_func, optimizer=optimizer)
#
#     return dnn

data = read_training_data()
num_epochs = 50
num_hidden_layers = 6
layer_size = 64
loss_func = keras.losses.mean_absolute_error

# test_proportion = .1
# layer_size_list = [32, 64, 128]
# loss_funcs = [keras.losses.mean_absolute_error, keras.losses.mean_squared_error]
# best_avg = np.inf
#
# for layer_size in layer_size_list:
#     for loss_func in loss_funcs:
#         num_epochs = 20
#         num_hidden_layers = 1
#         old_nn, old_training_loss, old_test_loss = train_dnn(data, num_epochs, num_hidden_layers, layer_size, loss_func)
#
#         nn, training_loss, test_loss = train_dnn(data, num_epochs, num_hidden_layers + 1, layer_size, loss_func)
#
#         while old_training_loss - training_loss > .01:
#             num_hidden_layers += 1
#             old_nn = nn
#             old_training_loss = training_loss
#             old_test_loss = test_loss
#
#             nn, training_loss, test_loss = train_dnn(data, num_epochs, num_hidden_layers + 1, layer_size, loss_func)
#
#         nn, training_loss, test_loss = train_dnn(data, num_epochs + 10, num_hidden_layers, layer_size, loss_func)
#
#         while old_training_loss - training_loss > .01 and old_test_loss - test_loss > .01:
#             num_epochs += 10
#             old_nn = nn
#             old_training_loss = training_loss
#             old_test_loss = test_loss
#
#             nn, training_loss, test_loss = train_dnn(data, num_epochs + 10, num_hidden_layers, layer_size, loss_func)
#
#         total_test_loss = 0
#         for _ in range(5):
#             nn, training_loss, test_loss = train_dnn(data, num_epochs, num_hidden_layers, layer_size, loss_func)
#
#             total_test_loss += test_loss
#
#         avg_test_loss = total_test_loss / 5
#
#         if avg_test_loss < best_avg:
#             best_avg = avg_test_loss
#             with open("best_nn_params.txt", "w") as f:
#                 f.write("num_epochs = " + str(num_epochs) + "\n")
#                 f.write("number of hidden layers = " + str(num_hidden_layers) + "\n")
#                 f.write("layer size = " + str(layer_size) + "\n")
#                 f.write("loss func = " + str(loss_func) + "\n")


# dnn, training_loss, test_loss = train_dnn(data, num_epochs, num_hidden_layers, layer_size, loss_func, test_proportion)
# print("Training Loss:", training_loss)
# print("Test Loss:", test_loss)

tot_training_loss = 0
tot_test_loss = 0

for i in range(10):
    dnn, training_loss, test_loss = train_dnn(data, num_epochs, num_hidden_layers, layer_size, loss_func)
    tot_training_loss += training_loss
    tot_test_loss += test_loss

avg_training_loss = tot_training_loss / 10
avg_test_loss = tot_test_loss / 10

print("Avg training loss:", avg_training_loss)
print("Avg test loss:", avg_test_loss)
