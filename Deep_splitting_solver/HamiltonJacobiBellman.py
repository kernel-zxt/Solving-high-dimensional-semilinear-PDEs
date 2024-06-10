import tensorflow as tf
import numpy as np
import os
import time
import shutil
from SplittingModel import simulate
#from DBFKSTModel import simulate

def phi(y):
    return tf.math.log((1 + tf.reduce_sum(tf.square(y), 1, keepdims=True)) / 2)

def f(x, y, z):
    return -tf.reduce_sum(z ** 2, axis=1, keepdims=True)


def sde(_d, n):
    sigma = np.sqrt(2.0)
    x_init = np.zeros(_d)
    sqrt_delta_t = np.sqrt(T / N)
    dw_sample = np.random.normal(size=[batch_size, _d, N+1]) * sqrt_delta_t  
    x_sample = np.zeros([batch_size, _d, N + 1])
    x_sample[:, :, 0] = np.ones([batch_size, _d]) * x_init
    for i in range(N):
        x_sample[:, :, i + 1] = x_sample[:, :, i] + sigma * dw_sample[:, :, i]         
    x_sample = tf.convert_to_tensor(x_sample)
    
    x = [tf.expand_dims(x_sample[:, :, n], axis=-1),
          tf.expand_dims(x_sample[:, :, n + 1], axis=-1)]
    x = tf.concat(x, axis=2)
    x_float32 = tf.cast(x, dtype=tf.float32)   
    return x_float32

d = 100
batch_size = 256
train_steps = 600
lr_boundaries = [400, 500]
lr_values = [0.1, 0.01, 0.001]

_file = open('HJB.csv', 'w')
_file.write('train_steps, loss, value, time, relative_error\n')

for train_steps in [500]:
     
    neurons = [d + 10, d + 10, 1]

    for N in [20]:

        T = N / 20.

        for run in range(1):

            path = '.../hjb'
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)

            t_0 = time.time()
            v_n,loss = simulate(T, N, d, sde, phi, f, neurons, train_steps, batch_size, lr_boundaries, lr_values, path)
            rel = np.abs((v_n - 4.5901)/4.5901)
            t_1 = time.time()

            _file.write('%i, %f, %f, %f, %f\n' % (train_steps, loss, v_n, t_1 - t_0, rel))
            _file.flush()

_file.close()
