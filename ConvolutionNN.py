import tensorflow as tf
import numpy as np

class ConvolutionNN():

    def __init__(self, n_classes=2, batch_size=1, keep_rate=.8):
        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.keep_rate = keep_rate
        self.keep_prob = tf.placeholder(tf.float32)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,100,1], padding='SAME')

    def maxpool2d(self, x):
        return tf.nn.max_pool(x, ksize=[1,1,1,1], strides=[1,1,1,1], padding='SAME')

    def neural_network_model(self, data):
        weights = {'W_conv1': tf.Variable(tf.random_normal([4, 100, 1, 30])),
                   'W_fc': tf.Variable(tf.random_normal([53 * 1 * 30, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, self.n_classes]))}

        biases = {'b_conv1': tf.Variable(tf.random_normal([30])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([self.n_classes]))}
        print tf.shape(data)
        x = tf.reshape(data, shape=[-1, 53, 100, 1])

        conv1 = tf.nn.relu(self.conv2d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = self.maxpool2d(conv1)

        fc = tf.reshape(conv1, [-1, 53 * 1 * 30])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out']) + biases['out']

        return output

    def train_neural_network(self, train_data, test_data):

        prediction = self.neural_network_model(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 10

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(len(train_data)/self.batch_size):
                    train_x = np.array(list(train_data[_ * self.batch_size:(_ + 1) * self.batch_size, 0]))
                    train_y = np.array(list(train_data[_ * self.batch_size:(_ + 1) * self.batch_size, 1]))
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: train_x, self.y: train_y})
                    epoch_loss += c
                print "Epoch", epoch, "completed out of ", hm_epochs, "loss: ", epoch_loss

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print "Accuracy", accuracy.eval({self.x: np.array(list(test_data[:, 0])), self.y: np.array(list(test_data[:, 1]))})

