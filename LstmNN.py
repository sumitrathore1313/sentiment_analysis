import tensorflow as tf
import numpy as np
tf.reset_default_graph()

class LstmNN():

    def __init__(self, batchSize = 100, lstmUnits = 64, numClasses = 2, iterations = 1):

        self.batch_size = batchSize
        self.lstmUnits = lstmUnits
        self.numClasses = numClasses
        self.iterations = iterations

        self.data = tf.Variable(tf.zeros([batchSize, 53, 100]),dtype=tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.int32)



    def train_neural_network(self, train_data, test_data):

        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstmCell, self.data, dtype=tf.float32)
        weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = self.iterations

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(len(train_data) / self.batch_size):
                    train_x = np.array(list(train_data[_ * self.batch_size:(_ + 1) * self.batch_size, 0]))
                    train_y = np.array(list(train_data[_ * self.batch_size:(_ + 1) * self.batch_size, 1]))
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: train_x, self.y: train_y})
                    epoch_loss += c
                print "Epoch", epoch, "completed out of ", hm_epochs, "loss: ", epoch_loss

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            a = 0
            for _ in range(len(test_data) / self.batch_size):
                test_x = np.array(list(test_data[_ * self.batch_size:(_ + 1) * self.batch_size, 0]))
                test_y = np.array(list(test_data[_ * self.batch_size:(_ + 1) * self.batch_size, 1]))
                a += accuracy.eval({self.x: test_x, self.y: test_y})
            print "Accuracy", a*100/(len(test_data) / self.batch_size)



