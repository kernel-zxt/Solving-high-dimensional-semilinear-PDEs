# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:51:46 2021

@author: kernel
"""
#import torch
import logging
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import math
import torch
import os
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.dates as mdate
from matplotlib.pyplot import rcParams 
import matplotlib.ticker as ticker
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        
        self.model = NonsharedModel(config, bsde)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
        self.no_improvement_count = 0

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        for step in range(self.net_config.num_iterations+1):
            if step % self.net_config.logging_frequency == 0:
                y_t0,loss = self.loss_fn(valid_data, training=False)
                loss = loss.numpy()
                y_t0 = tf.reduce_mean(y_t0)
                y_terminal = y_t0
                """append 一下"""
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_terminal, elapsed_time])
                if self.net_config.verbose:
                    logging.info("step: %5u,    loss: %.4e  Y0: %.4e   elapsed time: %3u" % (
                        step, loss, y_t0, elapsed_time))
            self.train_step(self.bsde.sample(self.net_config.batch_size))
            #"""The reference for stopping training"""
            # if step > 3 and len(training_history) > 1 and loss > training_history[-2][1]:
            #     self.no_improvement_count += 1
            # else:
            #     self.no_improvement_count += 0
            # if self.no_improvement_count >= 1:
            #     break
        return np.array(training_history)

    
    def loss_fn(self, inputs, training):
        dw, x = inputs
        y_t0,delta,y_terminal = self.model.call(inputs, training) 
        y_t0 = tf.reduce_mean(y_t0)
        loss = tf.reduce_mean(delta)
        
        return y_t0,loss

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            y_t0,loss = self.loss_fn(inputs, training)
        # print(tf.shape(self.model.trainable_variables))
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        
        

        self.subnet_u = [FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval-1)]
    
       
    def call(self, inputs, training):
        dw, x = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval+1) * self.bsde.delta_t
        
        delta = 0


        self.sigma = self.bsde.sigma
        self.sigma = np.float64(self.sigma)

        xx = tf.convert_to_tensor(x[:, :, -1])
        with tf.GradientTape() as tape:             
            tape.watch(xx)
            y_terminal = self.bsde.g_tf(self.bsde.total_time, xx)
        
        grad_T = (tape.gradient(y_terminal, xx)) 
        grad_T = tf.multiply(self.sigma, grad_T) 
        
        y_terminal2 = y_terminal + self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[self.bsde.num_time_interval], x[:, :, -1], y_terminal, grad_T)) #通过递推式更新上一时刻的解
        
        """the data for the time step t_{N-1}"""
        y = y_terminal2
        delta_y = y - self.subnet_u[self.bsde.num_time_interval-2].call(x[:, :, self.bsde.num_time_interval-1], training)
        delta = delta + tf.square(delta_y)
        
        # print(tf.reduce_mean(y_terminal2))
        """the data for the time step t_{n} n=1,2...,N-2"""
        
        for t in range(self.bsde.num_time_interval-2,0 ,-1):
            z = self.subnet_u[t].net_f(x[:, :, t+1],training) 
            z = tf.multiply(self.sigma, z) 
            y = y + self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t+1], x[:, :, t+1], self.subnet_u[t].call(x[:, :, t ], training), z)) 
            
            delta_y = y - self.subnet_u[t-1].call(x[:, :, t ], training) 
            delta = delta + tf.square(delta_y)
             
        z = self.subnet_u[0].net_f(x[:, :, 1 ],training)
        z = tf.multiply(self.sigma, z)
        y_t0 = y + self.bsde.delta_t * self.bsde.f_tf(time_stamp[1], x[:, :, 1], self.subnet_u[0].call(x[:, :, 1 ], training), z) 
        
        return y_t0,delta,y_terminal
        
        

class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__()
        # dim = config.eqn_config.dim
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(1, activation=None))
           

    
    """u,1维"""
    def call(self, x,  training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        u = self.bn_layers[0](x,training)
        for i in range(len(self.dense_layers) - 1):
            u = self.dense_layers[i](x)
            u = self.bn_layers[i+1](x, training)
            u = tf.nn.relu(x)
        u = self.dense_layers[-1](x)
        
        return u
    
    def tensor_to_array(array_value):
        return array_value.numpy()
    
    
    def net_f(self, x,training):   
        x = tf.convert_to_tensor(x)              
        with tf.GradientTape() as tape:             
            tape.watch(x)
            u = self.call(x ,training)
       
        grad = (tape.gradient(u, x))
        del tape
        return grad
        
       