import tensorflow as tf
import numpy as np

class FeedForwardNN():

    def __init__(self, n_node_h1 = 100, n_node_h2 = 100, n_classes = 2, batch_size = 10, vector_size=100):
        self.n_node_h1 = n_node_h1
        self.n_node_h2 = n_node_h2
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')
        self.vector_size = vector_size

    def neural_network_model(self, data):
        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([self.vector_size, self.n_node_h1])),
                          'biases': tf.Variable(tf.random_normal([self.n_node_h1]))}
        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([self.n_node_h1, self.n_node_h2])),
                          'biases': tf.Variable(tf.random_normal([self.n_node_h2]))}
        output_layer = {'weights': tf.Variable(tf.random_normal([self.n_node_h2, self.n_classes])),
                          'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

        return output

    def train_neural_network(self, train_data, test_data):

        pridiction = self.neural_network_model(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pridiction, labels=self.y))
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

            correct = tf.equal(tf.argmax(pridiction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print "Accuracy", accuracy.eval({self.x: np.array(list(test_data[:, 0])), self.y: np.array(list(test_data[:, 1]))})



