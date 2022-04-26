# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:11:51 2022

@author: kki
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def array1():
    # 1차원 array
    t = np.array([0., 1., 2., 3., 4., 5., 6.])
    print(t.ndim) # 몇차원 array? rank
    print(t.shape) # array이가 어떻게 생겼냐?
    print(t[0], t[1], t[-1])
    print(t[2:5], t[4:-1])
    print(t[:2], t[3:])
# array1()

def array2():
    t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
    print(t,'\n',t.ndim)
    print(t.shape)
# array2()

def shape_rank_axis():
    t = tf.constant([1, 2, 3, 4])
    print(t.get_shape())
    
    t = tf.constant([[1, 2], [3, 4]])
    # tf.shape(t).eval()
    
    t = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10 ,11, 12]],
                     [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
    # tf.shape(t).eval()
shape_rank_axis()

def multi_vs_multiply():
    matrix1 = tf.constant([[1., 2.], [3., 4.]])
    matrix2 = tf.constant([[1.], [2.]])
    print("matrix 1 shape", matrix1.shape)
    print("matrix 2 shpae", matrix2.shape)
    # print(tf.matmul(matrix1, matrix2).eval())

# multi_vs_multiply()

def broadcasting():
    matrix1 = tf.constant([[1., 2.]])
    matrix2 = tf.constant(3.)
    print((matrix1 + matrix2))
    
# broadcasting()

def reduce_mean():
    tf.reduce_mean([1, 2], axis=0)
    x = [[1., 2.],
         [3., 4.]]
    tf.reduce_mean(x, axis = 0)
    tf.reduce_mean(x, axis = 1)
    tf.reduce_mean(x, axis = -1)
    
# reduce_mean()

def reduce_sum():
    x = [[1., 2.],
         [3., 4.]]
    
    tf.reduce_sum(x)
    tf.reduce_sum(x, axis = 0)
    tf.reduce_sum(x, axis = 1)
    tf.reduce_sum(x, axis = -1)
    tf.reduce_mean(tf.reduce_sum(x, axis=-1))

# reduce_sum()

def argmax(): # argmax는 가장 큰 값의 위치
    x = [[0, 1, 2],
         [2, 1, 0]]
    tf.argmax(x, axis = 0)
    tf.argmax(x, axis = 1)
    tf.argmax(x, axis = -1)
    
# argmax()

def reshape():
    t = np.array([[[0, 1, 2],
                  [3, 4, 5]],
                 
                 [[6, 7, 8],
                  [9, 10, 11]]])
    
    t.shape()
    tf.reshape(t, shape=[-1, 3])
    tf.reshape(t, shape=[-1, 1, 3])
    
    tf.squeeze([[0], [1], [2]])
    tf.expand_dims([0, 1, 2], 1)

# reshape()

def one_hot():
    tf.one_hot([[0], [1], [2], [0]], depth=3)
    t = tf.one_hot([[0], [1], [3], [0]], depth=3)
    tf.reshape(t, shape=[-1, 3])

# one_hot()

def cast():
    tf.cast([1.9, 2.2, 3.3, 4.9], tf.int32)
    tf.cast([True, False, 1 ==1, 0 ==1], tf.int32)

# cast()

def stack():
    x = [1, 4]
    y = [2, 5]
    z = [3, 6]
    
    tf.stack([x, y, z])
    
    tf.stack([x, y, z], axis=1)
# stack()

def ones_zeros():
    x = [[0, 1, 2],
         [2, 1, 0]]
    
    tf.ones_like(x)
    tf.zeros_like(x)
    
# ones_zeros()

def zip_():
    for x, y in zip([1, 2, 3], [4, 5, 6]):
        print(x, y)

    for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
        print(x, y, z)
    
# zip_()










