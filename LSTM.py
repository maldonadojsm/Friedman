# !/usr/bin/env python
# title           :LSTM
# description     :LSTM Model That Predicts Closing Price of Stock at T+1
# author          :Juan Maldonado
# date            :5/2/19
# version         :1.0
# usage           :python3 LSTM.py
# notes           :SEE README.txt for list of dependencies
# python_version  :3.6.5
# =================================================================================================================



import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error


#################################################

# PRE-PROCESSING

# Importing GE Dataset

# Build Frame
dataset = pd.read_csv("GE_1980_2019.csv", index_col=0)
data_frame = dataset.copy()
# Drop Missing Values
data_frame = data_frame.dropna()

data_frame = data_frame[['Open', 'High', 'Low', 'Close', '5MA', '8MA', '13MA', 'MACD']]


# Normalizing Features

def normalize(data_frame):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    # Opening Price
    data_frame['Open'] = min_max_scaler.fit_transform(data_frame['Open'].values.reshape(-1, 1))
    # High Day Price
    data_frame['High'] = min_max_scaler.fit_transform(data_frame['High'].values.reshape(-1, 1))
    # Lowest Day Price
    data_frame['Low'] = min_max_scaler.fit_transform(data_frame['Low'].values.reshape(-1, 1))
    # Closing Price
    data_frame['Close'] = min_max_scaler.fit_transform(data_frame['Close'].values.reshape(-1, 1))
    # 5 Day Moving Average
    data_frame['5MA'] = min_max_scaler.fit_transform(data_frame['5MA'].values.reshape(-1, 1))
    # 8 Day Moving Average
    data_frame['8MA'] = min_max_scaler.fit_transform(data_frame['8MA'].values.reshape(-1, 1))
    # 13 Day Moving Average
    data_frame['13MA'] = min_max_scaler.fit_transform(data_frame['13MA'].values.reshape(-1, 1))
    # Moving Average Convergence Divergence (MACD)
    data_frame['MACD'] = min_max_scaler.fit_transform(data_frame['MACD'].values.reshape(-1, 1))

    return data_frame


# Normalized GE Stock DataFrame


norm_stock_df = data_frame.copy()
norm_stock_df = normalize(norm_stock_df)

####################################################################

# SPLITTING DATASET (70% Train, 15% Validation, 15% Test)


validation_percentage = 15
test_percentage = 15

seq_length = 20


def splitData(stock, sequence_length):
    raw_data = stock.as_matrix()
    data = []
    for i in range(len(raw_data) - sequence_length):
        data.append(raw_data[i:i + sequence_length])
    data = np.array(data)
    # Validation Set
    valid_dset_size = int(np.round(validation_percentage/100 * data.shape[0]))
    # Testing Set
    test_dset_size = int(np.round(test_percentage / 100 * data.shape[0]))
    # Training Set
    train_dset_size = data.shape[0] - (valid_dset_size + test_dset_size)

    x_train = data[:train_dset_size, :-1, :]
    y_train = data[:train_dset_size, -1, :]

    x_validation = data[train_dset_size:train_dset_size + valid_dset_size, :-1, :]
    y_validation = data[train_dset_size:train_dset_size + valid_dset_size, -1, :]

    x_test = data[train_dset_size+valid_dset_size:, :-1, :]
    y_test = data[train_dset_size+valid_dset_size:, -1, :]

    return [x_train, y_train, x_validation, y_validation, x_test, y_test]


x_train, y_train, x_validation, y_validation, x_test, y_test = splitData(norm_stock_df, seq_length)

# DataSet Summary (Debugging)

# print("x_train.shape = ", x_train.shape)
# print("y_train.shape = ", y_train.shape)
# print("x_valid.shape = ", x_validation.shape)
# print("y_valid.shape = ", y_validation.shape)
# print("x_test.shape = ", x_test.shape)
# print("y_test.shape = ", y_test.shape)


###########################################################################

# NEURAL NETWORK CONSTRUCTION


# PARAMETERS (Will adjust constantly for the purpose of producing differing prediction results)

num_steps = seq_length - 1
num_inputs = 8
num_neurons = 40
num_outputs = 8
num_layers = 2
learning_rate = 0.001
batch_size = 10
num_epochs = 40
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

# Network will capture 2-D Matrix, Return 1-D Vector

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, num_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_outputs])


# Batch Processing

index_epoch = 0
perm_array = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

# Captures next batch of observations
def nextBatch(batch_size):
    global index_epoch, x_train, perm_array
    start = index_epoch
    index_epoch += batch_size
    # Begin next Epoch
    if index_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)
        start = 0
        index_epoch = batch_size
    end = index_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

# Layer Design: LSTM APPROACH


layers = [tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.elu) for layer in range(num_layers)]

RNN_layers = tf.contrib.rnn.MultiRNNCell(layers)
RNN_outputs, states = tf.nn.dynamic_rnn(RNN_layers, X, dtype=tf.float32)
stcK_RNN_outputs = tf.reshape(RNN_outputs, [-1, num_neurons])
stcK_outputs = tf.layers.dense(stcK_RNN_outputs, num_outputs)
outputs = tf.reshape(stcK_outputs, [-1, num_steps, num_outputs])

outputs = outputs[:, num_steps - 1, :]


# COST FUNCTION (Mean Squared Error)

loss = tf.reduce_mean(tf.square(outputs - y))


# LEARNING RATE OPTIMIZER

optimize = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimize_training = optimize.minimize(loss)


# TRAIN MODEL

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(int(num_epochs * train_set_size/batch_size)):
        x_batch, y_batch = nextBatch(batch_size)
        session.run(optimize_training, feed_dict={X: x_batch, y: y_batch})
        if i % int(train_set_size / batch_size) == 0:
            MSE_train = loss.eval(feed_dict={X: x_train, y: y_train})
            MSE_validation = loss.eval(feed_dict={X: x_validation, y: y_validation})
            print('%.2f epochs: MSE train/valid = %.6f/%.6f' % (i * batch_size / train_set_size, MSE_train, MSE_validation))

    y_validation_pred = session.run(outputs, feed_dict={X: x_validation})
    y_training_pred = session.run(outputs, feed_dict={X: x_train})
    y_testing_pred = session.run(outputs, feed_dict={X: x_test})


# ANALYZE & COMPARE PREDICTIONS TO TEST LABELS

compare = pd.DataFrame({'Column1': y_test[:, 3], 'Column2': y_testing_pred[:, 3]})
figure = plt.figure(figsize=(10, 5))
plt.title('LSTM : MSE  Tst: %.6f / Vld: %.6f | Epochs : %d | LRate : %.3f | Nrns: %d | Lyrs : %d | BSize : %d' %
          (mean_squared_error(y_test, y_testing_pred), mean_squared_error(y_validation, y_validation_pred), num_epochs, learning_rate, num_neurons, num_layers, batch_size))
plt.ylabel("Price")
plt.xlabel("Time")
plt.plot(compare['Column1'], color='red', label='Target')
plt.plot(compare['Column2'], color='blue', label='Prediction')
plt.legend()
#plt.show()

figure.savefig('LSTM/LSTMBestConfig' + '.png')