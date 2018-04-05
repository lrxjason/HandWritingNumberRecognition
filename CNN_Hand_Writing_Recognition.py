import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2


def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((140, 140, 3), np.uint8)


n_input = 784
n_output = 10
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
}


def conv_basic(_input, _w, _b, _keepratio):
    # INPUT
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    # CONV LAYER 1
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # _mean, _var = tf.nn.moments(_conv1, [0, 1, 2])
    # _conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0, 1, 0.0001)
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # CONV LAYER 2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    # _mean, _var = tf.nn.moments(_conv2, [0, 1, 2])
    # _conv2 = tf.nn.batch_normalization(_conv2, _mean, _var, 0, 1, 0.0001)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # VECTORIZE
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
           'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
           }
    return out


print("CNN READY")

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# FUNCTIONS

_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_pred, labels=y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()

# SAVER
save_step = 10
saver = tf.train.Saver(max_to_keep=3)

print("GRAPH READY")

do_train = 0
sess = tf.Session()
sess.run(init)

training_epochs = 10
batch_size = 128
display_step = 1
if do_train == 1:

    mnist = input_data.read_data_sets('data/', one_hot=True)
    trainimg = mnist.train.images
    trainlabel = mnist.train.labels
    testimg = mnist.test.images
    testlabel = mnist.test.labels
    print("MNIST ready")

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.9})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.}) / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})
            print(" Training accuracy: %.3f" % train_acc)
            test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio: 1.})
            print(" Test accuracy: %.3f" % test_acc)
    # Save Net
    saver.save(sess, "save_CNN_model/cnn_mnist")

    print("OPTIMIZATION FINISHED")

if do_train == 0:

    drawing = False
    mode = True
    ix, iy = -1, -1
    img = np.zeros((140, 140, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    saver.restore(sess, "save_CNN_model/cnn_mnist")
    prediction = tf.argmax(_pred, 1)

    # mnist = input_data.read_data_sets('data/', one_hot=True)
    # trainimg = mnist.train.images
    # trainlabel = mnist.train.labels
    # testimg = mnist.test.images
    # testlabel = mnist.test.labels
    # print("MNIST ready")
    # test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio: 1.})
    # print(" TEST ACCURACY: %.3f" % (test_acc))

    while 1:
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == 97:
            print('Image saved')
            cv2.imwrite('aaa.jpg', img)
            img = np.zeros((140, 140, 3), np.uint8)

            image = cv2.imread('aaa.jpg', 0)
            resized_image = cv2.resize(image, (28, 28))
            im = np.reshape(np.array(resized_image), (-1, 784))
            answer = prediction.eval(feed_dict={x: im, keepratio: 1.}, session=sess)
            print("prediction", answer)
