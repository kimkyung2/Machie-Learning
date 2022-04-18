# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:09:07 2022

@author: kki
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def file_multi():
    ### 데이터 읽엉오기 ####
    xy = np.loadtxt('data_test_score.csv', delimiter = ',', dtype = np.float32) 
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]
    
    # print(x_data.shape, x_data, len(x_data))
    # print(y_data.shape, y_data)
    
    x = tf.placeholder(tf.float32, shape = [None, 3])
    y = tf.placeholder(tf.float32, shape = [None, 1])
    
    W = tf.Variable(tf.random_normal([3, 1]), name='weight') # 3은 들어오는 x 값, 1은 들어오는 y 값
    b = tf.Variable(tf.random_normal([1]), name = 'bias') # y의 값과 같음
    
    hypothesis = tf.matmul(x, W) + b
    
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
    train = optimizer.minimize(cost)
    
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x: x_data, y: y_data})
        if step % 10 == 0:
            print(step, "cost: ", cost_val, "Prediction: ", hy_val)
            
    print("your score will be ", sess.run(hypothesis, feed_dict = {x: [[100, 70, 101]]}))
    print("opter scores will be ", sess.run(hypothesis, feed_dict={x: [[60, 70, 110], [90, 100, 80]]}))
# file_multi()

def file_multi_2():
    filename_queue = tf.train.string_input_producer(['data_test_score.csv'], shuffle=False, name ='filename_queue')
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    
    record_defaults = [[0.], [0.], [0.], [0.]]
    xy = tf.decode_csv(value, record_defaults = record_defaults)
    
    train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
    
    x = tf.placeholder(tf.float32, shape = [None, 3])
    y = tf.placeholder(tf.float32, shape = [None, 1])
    
    W = tf.Variable(tf.random_normal([3, 1]), name ='weight')
    b = tf.Variable(tf.random_normal([1]), name = 'bias')
    
    hypothesis = tf.matmul(x, W) + b
    
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
    train = optimizer.minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
    for step in range(2001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x: x_batch, y: y_batch})
        if step % 10 == 0:
            print(step, "cost: ", cost_val, "Prediction: ", hy_val)
            
    coord.request_stop()
    coord.join(threads)
    
# file_multi_2()

import tensorflow as tf

def tensorflow2_0():
    xy = np.loadtxt('data_test_score.csv', delimiter = ',', dtype = np.float32)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]
    
    print(x_data, 'nx_data shape: ', x_data.shape)
    print(y_data, 'ny_data shape: ', y_data.shape)
    
    tf.model = tf.keras.Sequential()
    # 활성화 기능을 별도의 계층으로 추가 할 필요 없음, Dense() 층의 인수로 추가
    tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation ='linear'))
    tf.model.summary()
    
    tf.model.compile(loss='mse', optimizer = tf.keras.optimizers.SGD(lr=1e-5))
    history = tf.model.fit(x_data, y_data, epochs = 2000)
    
    print("Your score will be ", tf.model.predict([[100, 70, 101]]))
    print("Other scores will be ", tf.model.predict([[60, 70, 110],[ 90, 100, 80]]))
    
    
tensorflow2_0()
































