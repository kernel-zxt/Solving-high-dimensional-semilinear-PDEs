import tensorflow as tf
import numpy as np
import os
import time
import shutil
from SplittingModel import simulate
#from DBFKSTModel import simulate

def phi(y):
    return tf.reduce_min(y, axis=1, keepdims=True)


def f(x, y, z):
    #return -(1. - delta) * tf.minimum(tf.maximum((y - v_h) * (gamma_h - gamma_l) / (v_h - v_l) + gamma_h, gamma_l), gamma_h) * y - R * y
    slope = (gamma_h - gamma_l) / (v_h - v_l)
    piecewise_linear = tf.nn.relu(
            tf.nn.relu(y - v_h) * slope + gamma_h - gamma_l) + gamma_l
    return (-(1 - delta) * piecewise_linear - R) * y

def sde(_d, n):
    mu_bar = 0.02
    sigma = 0.2
    delta_t = T / N
    x_init = np.ones( _d) * 100.0
    sqrt_delta_t = np.sqrt(T / N)
    dw_sample = np.random.normal(size=[batch_size, _d, N+1]) * sqrt_delta_t
    x_sample = np.zeros([batch_size, _d, N + 1])
    x_sample[:, :, 0] = np.ones([batch_size, _d]) * x_init
    for i in range(N):
        x_sample[:, :, i + 1] = (1 + mu_bar * delta_t) * x_sample[:, :, i] + (
                sigma * x_sample[:, :, i] * dw_sample[:, :, i])
    x_sample = tf.convert_to_tensor(x_sample)
    
    x = [tf.expand_dims(x_sample[:, :, n], axis=-1),
          tf.expand_dims(x_sample[:, :, n + 1], axis=-1)]
    x = tf.concat(x, axis=2)
    x_float32 = tf.cast(x, dtype=tf.float32) 
    
    return x_float32


def sde_loop(_d, n):
    xi = tf.ones((batch_size, _d)) * 100.

    def loop(_n, _x0, _x1):
        _x0 = _x1
        _x1 = _x1 * (1. + mu_bar * T / N + sigma_bar * tf.random_normal((batch_size, _d), stddev=np.sqrt(T / N)))
        return _n + 1, _x0, _x1

    _, x0, x1 = tf.while_loop(lambda _n, _x0, _x1: _n <= n, loop, (tf.constant(0), xi, xi))

    return tf.stack([x0, x1], axis=2)

d = 100
N, T = 20, 1.
delta, R = 2. / 3., 0.02
mu_bar, sigma_bar = 0.02, 0.2
v_h, v_l = 50., 70.
gamma_h, gamma_l = 0.2, 0.02
lr_values = [0.1, 0.01, 0.001]
_file = open('nonlinear_BS.csv', 'w')
_file.write('train_steps, loss, value, time, relative_error\n')

#for train_steps in [0,100,200,600,1000,2000,5000,10000]:
for train_steps in [600]:
    neurons = [d + 10, d + 10, 1] #if d > 100 else [d + 50, d + 50, 1]
    batch_size = 256 #if d > 100 else 4096
    #train_steps = 600 #if d > 100 else 3000
    lr_boundaries = [2500, 3250] #if d > 100 else [2500, 2750]

    for run in range(1 ):

        path = '.../bs'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()
        v_n,loss = simulate(T, N, d, sde_loop if d > 100 else sde, phi, f, neurons, train_steps, batch_size, lr_boundaries,
                       lr_values, path)
        rel = np.abs((v_n - 57.3)/57.3)
        t_1 = time.time()

        _file.write('%i, %f, %f, %f, %f\n' % (train_steps, loss, v_n, t_1 - t_0, rel))
        _file.flush()

_file.close()
