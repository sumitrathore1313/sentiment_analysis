import tensorflow as tf
import numpy as np
import datetime
tf.reset_default_graph()

class LstmNN():

    def __init__(self, batchSize = 24, lstmUnits = 64, numClasses = 2, iterations = 100000):

        self.batch_size = batchSize
        self.lstmUnits = lstmUnits
        self.numClasses = numClasses
        self.iterations = iterations

        self.x = tf.Variable(tf.zeros([batchSize, 53, 100]),dtype=tf.float32)
        self.y = tf.placeholder(tf.float32, [batchSize, numClasses])


    def train_neural_network(self, train_data, test_data):

        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstmCell, self.x, dtype=tf.float32)
        weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        tf.summary.scalar('Loss', cost)
        tf.summary.scalar('Accuracy', accuracy)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        for i in range(self.iterations):

            #Next Batch of reviews
            nextBatch, nextBatchLabels = getTrainBatch();
            sess.run(optimizer, {self.x: nextBatch, self.y: nextBatchLabels})

            #Write summary to Tensorboard
            if (i % 50 == 0):
                summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                writer.add_summary(summary, i)

            #Save the network every 10,000 training iterations
            if (i % 10000 == 0 and i != 0):
                save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
                print("saved to %s" % save_path)
        writer.close()

        hm_epochs = 10

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



            print "Accuracy", accuracy.eval(
                {self.x: np.array(list(test_data[:, 0])), self.y: np.array(list(test_data[:, 1]))})



