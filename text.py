import tensorflow as tf
import cPickle
import os.path
import numpy as np

def load_data():
    f = open('data/ready_polarity_dataset.p', 'rb')
    revs, W, W2, word_idx_map, vocab = cPickle.load(f)
    data = {'revs': revs, 'w': W, 'w2': W2, 'word_idx_map': word_idx_map, 'vocab': vocab}
    return data


def make_sen2vec(data):
    sen2vec = []
    for sentence in data['revs']:
        temp = np.zeros(300, dtype=float)
        text = sentence['text']
        output = sentence['y']
        for word in text.split():
            id = data['word_idx_map'][word]
            temp = np.add(temp , data['w'][id])
        temp = temp/sentence['num_words']
        if output == 1:
            output = [0,1]
        else:
            output = [1,0]

        sen2vec.append([temp, output])
    f = open('data/sen2vec.p', 'wb')
    cPickle.dump(sen2vec, f)
    f.close()


n_node_h1 = 200
n_node_h2 = 200

n_classes = 2
batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([300, n_node_h1])),
                      'biases': tf.Variable(tf.random_normal([n_node_h1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_node_h1, n_node_h2])),
                      'biases': tf.Variable(tf.random_normal([n_node_h2]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_node_h2, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
    global y, test_data, train_data
    pridiction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pridiction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            train_data = np.array(train_data)
            train_x = list(train_data[:, 0])
            train_y = list(train_data[:, 1])
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print "Epoch", epoch, "completed out of ", hm_epochs, "loss: ", epoch_loss

        correct = tf.equal(tf.argmax(pridiction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print "Accuracy", accuracy.eval({x: np.array(list(test_data[:, 0])), y: np.array(list(test_data[:, 1]))})

# make sentence vector for each sentence
if not os.path.isfile('data/sen2vec.p'):
    # load data from pickle file
    loaded_data = load_data()
    make_sen2vec(loaded_data)

f = open('data/sen2vec.p', 'rb')
data = cPickle.load(f)
train_data = data[0:9000]
test_data = np.array(data[9000:])

train_neural_network(x)



