# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:15:39 2022

@author: kki
"""
import numpy as np
import matplotlib.pyplot as plt

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data , 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def RNN_stock():
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
    
    timesteps = seq_length = 7
    data_dim = 5
    output_dim = 1
    hidden_dim = 10
    
    # Open ,High, low, Close, Volume
    xy = np.loadtxt('데이터넣기', delimiter=',')
    xy = xy[::-1] # 시간순으로 정렬하기 위해 뒤집기
    xy = MinMaxScaler(xy)
    x = xy # 
    y = xy[:, [-1]]
    
    datax = []
    datay = []
    
    for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length]
        print(_x, "->", _y)
        datax.append(_x)
        datay.append(_y)
        
    # Training and test dataset으로 분류
    train_size = int(len(datay) * 0.7)
    test_size = len(datay) - train_size
    trainX, testX = np.array(datax[0:train_size]), np.array(datax[train_size:len(datax)])
    trainY, testY = np.array(datay[0:train_size]), np.array(datay[train_size:len(datay)])
    
    # input placeholders
    x = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    y = tf.placeholder(tf.float32, [None, 1])
    
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple = True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn = None)
    
    # cost/loss
    loss = tf.reduce_sum(tf.square(y_pred - y))
    
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(1000):
        _, l = sess.ruun([train, loss], feed_dict = {x: trainX, y:trainY})
        print(i, l)
        
    testPredict = sess.run(y_pred, feed_dict = {x: testX})
    

    plt.plot(testY)
    plt.plot(testPredict)
    plt.show()
# RNN_stock()

def build_dataset(time_series, seq_length):
    datax = []
    datay = []
    
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i+seq_length, :]
        _y = time_series[i + seq_length, [-1]]
        print(_x, "->", _y)
        datax.append(_x)
        datay.append(_y)
    return np.array(datax), np.array(datay)

def RNN_stock_2():
    import tensorflow as tf
    import os
    
    
    seq_length = 7
    data_dim = 5
    hidden_dim = 10
    output_dim = 1
    learning_rate = 0.01
    iterations = 500
    
    xy = np.loadtxt('', delimiter = ',')
    xy = xy[::-1] 
    
    # train/test split
    train_size = int(len(xy) * 0.7)
    train_set = xy[0:train_size]
    test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence
    
    # Scale each
    train_set = MinMaxScaler(train_set)
    test_set = MinMaxScaler(test_set)
    
    trainX, trainY = build_dataset(train_set, seq_length)
    testX, testY = build_dataset(test_set, seq_length)
    
    x = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    y = tf.placeholder(tf.float32, [None, 1])
    
    # build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_dim, state_is_tuple = True, activation = tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn = None)
    
    loss = tf.reduce_sum(tf.square(y_pred - y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
    
        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                                    x: trainX, y: trainY})
            print("[step: {}] loss: {}".format(i, step_loss))
    
        # Test step
        test_predict = sess.run(y_pred, feed_dict={x: testX})
        rmse_val = sess.run(rmse, feed_dict={
                        targets: testY, predictions: test_predict})
        print("RMSE: {}".format(rmse_val))
    
        # Plot predictions
        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.show()

    
RNN_stock_2()