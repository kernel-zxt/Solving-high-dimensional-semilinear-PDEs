import tensorflow as tf
import numpy as np
import os
import time
import shutil
from SplittingModel import simulate
#from DBFKSTModel import simulate


def phi(y):
    return 0.5 / (1 + 0.2 * tf.reduce_sum(tf.square(y), 1, keepdims=True))

def f(x, y, z):
    return y - y ** 3


def sde(_d, n):
    x = [tf.random_normal([batch_size, _d, 1], stddev=np.sqrt(2. * n * T / N)),
         tf.random_normal([batch_size, _d, 1], stddev=np.sqrt(2. * T / N))]
    return tf.cumsum(tf.concat(x, axis=2), axis=2)

d = 100
N, T = 20, 3. / 10.
batch_size = 256
#train_steps = 800
lr_boundaries = [300, 400]
lr_values = [0.1, 0.01, 0.001]

_file = open('AllenCahn.csv', 'w')
_file.write('train_steps, loss, value, time, relative_error\n')

for train_steps in [800]:

    neurons = [d + 10, d + 10, 1]

    for run in range(5):

        path = '.../allencahn'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()
        v_n, loss = simulate(T, N, d, sde, phi, f, neurons, train_steps, batch_size,
                       lr_boundaries, lr_values, path)
        rel = np.abs((v_n - 0.052802)/0.052802)
        t_1 = time.time()

        _file.write('%i, %f, %f, %f, %f\n' % (train_steps, loss, v_n, t_1 - t_0, rel))

_file.close()
