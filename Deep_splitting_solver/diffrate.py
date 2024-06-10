import tensorflow as tf
import numpy as np
import os
import time
import shutil
from SplittingModel import simulate
#from DBFKSTModel import simulate

def phi(y):
    temp = tf.reduce_max(y, 1, keepdims=True)
    return tf.maximum(temp - 120, 0) - 2 * tf.maximum(temp - 150, 0)
    #return tf.reduce_min(y, axis=1, keepdims=True)


def f(x, y, z):
    temp = tf.reduce_sum(z, 1, keepdims=True) / sigma_bar
    return -rl * y - (mu_bar - rl) * temp + (
            (rb - rl) * tf.maximum(temp - y, 0))

def sde(_d, n):
    # y = [tf.zeros((batch_size, _d)) * 100.]
    # for _n in range(n + 1):
    #     y.append(y[-1] * (1. + mu_bar * T / N + sigma_bar * tf.random_normal((batch_size, _d), stddev=np.sqrt(T / N))))
    # return tf.stack(y[n:n + 2], axis=2)
    sigma = 0.2
    mu_bar = 0.06
    delta_t = T / N
    x_init = np.ones(_d) * 100
    sqrt_delta_t = np.sqrt(T / N)
    dw_sample = np.random.normal(size=[batch_size, _d, N+1]) * sqrt_delta_t  
    x_sample = np.zeros([batch_size, _d, N + 1])
    x_sample[:, :, 0] = np.ones([batch_size, _d]) * x_init
    factor = np.exp((mu_bar-(sigma**2)/2)*delta_t)
    for i in range(N):
        x_sample[:, :, i + 1] = (factor * np.exp(sigma * dw_sample[:, :, i])) * x_sample[:, :, i]   
    x_sample = tf.convert_to_tensor(x_sample)
    
    x = [tf.expand_dims(x_sample[:, :, n], axis=-1),
          tf.expand_dims(x_sample[:, :, n + 1], axis=-1)]
    x = tf.concat(x, axis=2)
    x_float32 = tf.cast(x, dtype=tf.float32) 
    return x_float32



def sde_loop(_d, n):
    xi = tf.zeros((batch_size, _d)) * 100.

    def loop(_n, _x0, _x1):
        _x0 = _x1
        _x1 = _x1 * (1. + mu_bar * T / N + sigma_bar * tf.random_normal((batch_size, _d), stddev=np.sqrt(T / N)))
        return _n + 1, _x0, _x1

    _, x0, x1 = tf.while_loop(lambda _n, _x0, _x1: _n <= n, loop, (tf.constant(0), xi, xi))

    return tf.stack([x0, x1], axis=2)

d = 100
N, T = 20, 0.5
mu_bar, sigma_bar = 0.06, 0.2
rl = 0.04
rb = 0.06
lr_values = [0.1, 0.01, 0.001]
_file = open('diffrate.csv', 'w')
_file.write('train_steps, loss, value, time, relative_error\n')

for train_steps in [2000]:
#for train_steps in [100,200,300,600,1000,2000,5000,10000]:
    neurons = [d + 10, d + 10, 1] #if d > 100 else [d + 50, d + 50, 1]
    batch_size = 256 
    lr_boundaries = [1800, 2200] #if d > 100 else [2500, 2750]

    for run in range(1):

        path = 'C:/Users/kernel/Desktop/paper_code/paper1_code/Deep_splitting_solver/diffrate'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()
        v_n, loss = simulate(T, N, d, sde_loop if d > 100 else sde, phi, f, neurons, train_steps, batch_size, lr_boundaries,
                       lr_values, path)
        rel = np.abs((v_n - 21.299)/21.299)
        t_1 = time.time()
        
        _file.write('%i, %f, %f, %f, %f\n' % (train_steps, loss, v_n, t_1 - t_0, rel))
        _file.flush()

_file.close()
