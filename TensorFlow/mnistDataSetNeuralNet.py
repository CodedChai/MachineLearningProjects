# mnist dataset is 60k handwritten digits(0-9) and all 28x28 pixels
# There's also 10k testing samples(all unique)

'''
input data > weight > hidden layer 1 (activation function) > weights > hidden layer 1
(activation function) > weights > output layer
This is a feed forward NN because you just keep passing the data straight through

compare output to intended output using cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)
    Goes backwards and manipulates the weights which is called backpropogation

feed forward + backprop = epoch

Biases added in after weights.
Biases mostly used if all input data is zero.
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
One hot can be useful for multiclass classifications
10 classes, (0-9)
So one hot will output like this
    0 = [1,0,0,0,0,0,0,0,0,0]
    1 = [0,1,0,0,0,0,0,0,0,0]
    ...
    9 = [0,0,0,0,0,0,0,0,0,1]

'''
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000

n_classes = 10
batch_size = 1000   # Only 100 batches at a time so not so much is in memory

# height x width
x = tf.placeholder('float', [None, 28*28])     # Input Data: Squash the image so doesn't have to be matrix
y = tf.placeholder('float')     # Label

def neural_network_model(data):
    # Create variables for our layers
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([28*28, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)   # Rectified linear = threshold function

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # Can change learning rate if you want but default is usually good
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Cycles of feed forward + backprop
    hm_epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # For loops train data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # Chunks through the data set for you
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)