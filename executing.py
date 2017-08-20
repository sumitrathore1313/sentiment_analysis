import DataPreprocessing as dp
import GloveVector as gv
import cPickle as pic
import os.path
import SentenceVector as sv
import random
#import FeedForwardNN as NNM
#import ConvolutionNN as CNN
import LstmNN as RNN
import numpy as np


positive_file = "data/full_positive.txt"
negative_file = "data/full_negative.txt"
glove_file = "/home/sumit/stanford-nlp/glove/wiki_GVector/glove.6B.100d.txt"
glove_vocab_file = "/home/sumit/stanford-nlp/glove/wiki_GVector/vocab.txt"
word2vec_file = "data/word2vec.p"
sen2vec_file = "data/sen2vec2D.p"

if not os.path.isfile(word2vec_file):
    # get vocablory of all file
    vocab = dp.get_vocab(positive_file)
    vocab += dp.get_vocab(negative_file)

    # get the glove vector
    known_vocab, unknown_vocab = gv.get_unknown_vocab(glove_vocab_file, vocab)
    Word2vec, vocab, ivocab = gv.get_wiki_glove_vector(glove_file, known_vocab)
    Word2vec, vocab, ivocab = gv.get_unknown_vec(unknown_vocab,Word2vec, vocab, ivocab)
    pic.dump([Word2vec, vocab, ivocab ], open(word2vec_file, "wb"))
    print "word2vec created"

if sen2vec_file == "data/sen2vec.p":
    if not os.path.isfile(sen2vec_file):
        # get sentence vector from file
        sen2vec =  sv.get_avg_sen2vec(positive_file, word2vec_file, [1,0])
        sen2vec += sv.get_avg_sen2vec(negative_file, word2vec_file, [0,1])
        random.shuffle(sen2vec)
        pic.dump(sen2vec, open(sen2vec_file, "wb"))
        print "sen2vec created"

if sen2vec_file == "data/sen2vec2D.p":
    if not os.path.isfile(sen2vec_file):
        a = dp.get_max_senLen(positive_file)
        b = dp.get_max_senLen(negative_file)
        m = max(a,b)
        sen2vec = sv.get_2D_sen2vec(positive_file, word2vec_file, [1, 0], m)
        sen2vec += sv.get_2D_sen2vec(negative_file, word2vec_file, [0, 1], m)
        random.shuffle(sen2vec)
        pic.dump(sen2vec, open(sen2vec_file, "wb"))
        print "sen2vec created"


# load sentence vector for training and testing
with open(sen2vec_file, "r") as f:
    sen2vec = pic.load(f)

train_data = np.array(sen2vec[0:9000])
test_data = np.array(sen2vec[9000:])
# running neural network model
print "traning start"
#my_model = CNN.ConvolutionNN()   #use sen2vec2D instead of sen2vec
#my_model = NNM.FeedForwardNN() #use sen2vec
my_model = RNN.LstmNN()
my_model.train_neural_network(train_data, test_data)


