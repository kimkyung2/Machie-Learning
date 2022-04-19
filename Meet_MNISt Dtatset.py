# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:03:04 2022

@author: kki
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import random
tf.set_random_seed(777)
# from tensorflow.examples.tutorials.mnist import input_data 

# 이 부분은 tensorflow가 낮은 버전이라서 2.x에서는 안돌아감,,, 이유 찾아봐야함
def mnist():
    # mnist = input_data.read_data.sets("MNIST_data/", one_hot = True)
    
    nb_classes = 10
    
    # MNIST data image of shape 28*28 = 784
    x = tf.placeholder(tf.float32, [None, 784])
    # 0-9 digits recognition = 10 classes 
    y = tf.placeholder(tf.float32, [None, nb_classes])
    
    W = tf.Variable(tf.random_normal([784, nb_classes]))
    b = tf.Variable(tf.random_normal([nb_classes]))
    
    # softmax
    hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
    
    # test model
    is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(y, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    # 전체 데이터 셋을 다 훈련시킨 것을 1 epoch이라고 함
    training_epochs = 15
    # 몇개씩 잘라서 훈련시킬지
    batch_size = 100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_xs, y: batch_ys})
                avg_cost += c / total_batch
                
            print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
# sample image show and prediction    
    r = random.randint(0, mnist.test.num_examples -1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict = {x: mnist.test.images[r:r+1]}))
    
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap = 'Greys', interpolation='nearest')
    plt.show
# mnist()


def mnist2_0():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(len(x_train), len(y_train), x_train.shape, y_train.shape)
    print(len(x_test), len(y_test), x_test.shape, y_test.shape)
    
    x_train, x_test = x_train/255.0, x_test/255.0   #Feature scaling 적용
    
    nb_classes = 10;
    
    x_train_new = x_train.reshape(len(x_train), 784)         #60000 * 784 배열로 변경 - 한행당 이미지 하나 
    y_train_new = np.zeros((len(y_train), nb_classes))       #60000 * 10 배열 생성
    for i in range(len(y_train_new)):                        
      y_train_new[i,y_train[i]] = 1                          #one-hot encoding 
    
    x_test_new = x_test.reshape(len(x_test), 784)         #60000 * 784 배열로 변경 - 한행당 이미지 하나 
    y_test_new = np.zeros((len(y_test), nb_classes))       #60000 * 10 배열 생성
    for i in range(len(y_test_new)):                        
      y_test_new[i,y_test[i]] = 1                          #one-hot encoding 
    
    # MNIST data image of shape 28 * 28 = 784
    X = tf.placeholder(tf.float32, [None, 784])
    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, nb_classes])   #6만개의 학습에 대한 10개의 가설 결과
    
    W = tf.Variable(tf.random_normal([784, nb_classes]))  #가설이 10개이고 가설별로 784개의 weigh을 가짐, 즉 7840개의 w
    b = tf.Variable(tf.random_normal([nb_classes]))       #가설이 10개니 가설의 b도 10
    
    # Hypothesis (using softmax)
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)  #60000 x 10 행렬 - 행별로 열의 값을 확율로 바꿈 
    
    #cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    
    # Test model
    is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    # parameters
    training_epochs = 15  #traing을 몇번 돌릴것인지
    batch_size = 100  #한번에 몇건씩 읽은것인지
    total_batch = int(len(x_train_new) / batch_size)
    
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            
            for i in range(total_batch):
                #print (epoch,batch_size )
                batch_xs = x_train_new[(epoch * batch_size):(epoch + 1) * batch_size]     
                batch_ys = y_train_new[(epoch * batch_size):(epoch + 1) * batch_size]
    
                _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += cost_val / total_batch
                
            print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
    
    
        # Test the model using test sets
        print(
            "Accuracy: ",
            accuracy.eval(
                session=sess, feed_dict={X: x_test_new, Y: y_test_new}
            ),
        )
    
        # Get one and predict
        random_idx = random.randrange(1,10000)
        print ("random_idx : ", random_idx)
        print(
            "Prediction: ",
            sess.run(tf.argmax(hypothesis, 1), feed_dict={X: x_test_new[random_idx : random_idx + 1]}),
        )
    
        plt.imshow(
            x_test_new[random_idx : random_idx + 1].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
        plt.show()
mnist2_0()





