import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

elif event == cv2.EVENT_LBUTTONUP:
    drawing = False
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((140, 140, 3), np.uint8)

# NETWORK TOPOLOGIES
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

# INPUTS AND OUTPUTS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# NETWORK PARAMETERS
stddev = 0.1
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print("NETWORK READY")

def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    return (tf.matmul(layer_2, _weights['out']) + _biases['out'])

# PREDICTION
pred = multilayer_perceptron(x, weights, biases)

# LOSS AND OPTIMIZER
cost = tf.reduce_sum(np.square(pred - y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

# INITIALIZER
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")

saver = tf.train.Saver()


training_epochs = 50
batch_size      = 64
display_step    = 10

do_train = 0
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)

if do_train == 1:
    mnist = input_data.read_data_sets('data/', one_hot=True)
    trainimg = mnist.train.images
    trainlabel = mnist.train.labels
    testimg = mnist.test.images
    testlabel = mnist.test.labels
    print("MNIST ready")
    
    # OPTIMIZE
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # ITERATION
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            sess.run(optm, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
        avg_cost = avg_cost / total_batch
        # DISPLAY
        if (epoch+1) % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            feeds = {x: batch_xs, y: batch_ys}
            train_acc = sess.run(accr, feed_dict=feeds)
            print ("TRAIN ACCURACY: %.3f" % (train_acc))
            feeds = {x: mnist.test.images, y: mnist.test.labels}
            test_acc = sess.run(accr, feed_dict=feeds)
            print ("TEST ACCURACY: %.3f" % (test_acc))
    
    saver_path = saver.save(sess, "save_BPNN_model/BPNN_model")
    print ("OPTIMIZATION FINISHED")

if do_train == 0:
    drawing = False
    mode = True
    ix, iy = -1, -1
    img = np.zeros((168, 140, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    saver.restore(sess, "save_BPNN_model/BPNN_model")
    prediction = tf.argmax(pred, 1)
    
    while(1):
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == 97:
            print('Image saved')
            cv2.imwrite('aaa.jpg', img)
            img = np.zeros((168, 140, 3), np.uint8)
            
            image = cv2.imread('aaa.jpg', 0)
            resized_image = cv2.resize(image, (28, 28))
            im = np.reshape(np.array(resized_image), (-1, 784))
            answer = prediction.eval(feed_dict={x: im}, session=sess)
            print("prediction", answer)







