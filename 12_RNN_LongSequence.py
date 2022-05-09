# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:25:40 2022

@author: kki
"""

import numpy as np



def RNN_LS():
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    sample = "if you want you"
    idx2char = list(set(sample)) # index -> char
    char2idx = {c : i for i, c in enumerate(idx2char)} # char -> idx
    
    # hyper parameters
    dic_size = len(char2idx)  # RNN input size (one hot size)
    rnn_hidden_size = len(char2idx)  # RNN output size
    num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
    batch_size = 1  # one sample data, one batch
    sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
    learning_rate = 0.1
    
    sample_idx = [char2idx[c] for c in sample]
    x_data = [sample_idx[:-1]] # x data sample(0~n-1) hello : hell
    y_data = [sample_idx[1:]] # y data sample(1~n) hello : ello
    
    x = tf.placeholder(tf.int32, [None, sequence_length]) # x data
    y = tf.placeholder(tf.int32, [None, sequence_length]) # y label
    
    x_one_hot = tf.one_hot(x, num_classes) # one hot : 1 -> 0 1 0 0 0 0 0 0 
    
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = rnn_hidden_size, state_is_tuple = True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state = initial_state, dtype = tf.float32)
    weights = tf.ones([batch_size. sequence_length])
    
    # Compute sequence cost/loss
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets = y, weights = weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    prediction = tf.argmax(outputs, axis = 2)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3000):
            l, _ = sess.run([loss, train], feed_dict = {x : x_data, y : y_data})
            result = sess.run(prediction, feed_dict={x: x_data})
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print(i, "loss: ", l, "Prediction: ", ''.join(result_str))
# RNN_LS()


def RNN_LS_2():
    import tensorflow as tf
    sample = "if you want you"
    idx2char = list(set(sample)) # index -> char
    char2idx = {c : i for i, c in enumerate(idx2char)} # char -> idx
    
    # hyper parameters
    dic_size = len(char2idx)  # RNN input size (one hot size)
    rnn_hidden_size = len(char2idx)  # RNN output size
    num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
    batch_size = 1  # one sample data, one batch
    sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
    learning_rate = 0.1
    
    sample_idx = [char2idx[c] for c in sample]
    x_data = [sample_idx[:-1]] # x data sample(0~n-1) hello : hell
    y_data = [sample_idx[1:]] # y data sample(1~n) hello : ello
    
    x = tf.Variable(tf.ones(shape = [None, sequence_length]), dtype = tf.int32)  # X data
    y = tf.Variable(tf.ones(shape = [None, sequence_length]), dtype = tf.int32)  # Y label
    x_one_hot = tf.one_hot(x, num_classes) # one hot : 1 -> 0 1 0 0 0 0 0 0 
    x_for_softmax = tf.reshape(x_one_hot, [-1, rnn_hidden_size])
    
    # softmax layer (rnn_hidden_size -> num_classes)
    softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
    softmax_b = tf.get_variable("softmax_b", [num_classes])
    outputs = tf.matmul(x_for_softmax, softmax_w) + softmax_b
    
    # expend the data (revive the batches)
    outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
    weights = tf.ones([batch_size, sequence_length])
    
    # Compute sequence cost/loss
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets = y, weights = weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    prediction = tf.argmax(outputs, axis = 2)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3000):
            l, _ = sess.run([loss, train], feed_dict = {x : x_data, y : y_data})
            result = sess.run(prediction, feed_dict={x: x_data})
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print(i, "loss: ", l, "Prediction: ", ''.join(result_str))
            
RNN_LS_2()