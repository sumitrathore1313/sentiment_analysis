from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import io
import cPickle as pic
import numpy as np


tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

def get_avg_sen2vec(fname, word2vec_fname, output):
    sen2vec = []
    with open(word2vec_fname, "r") as f:
        word2vec, vocab, ivocab = pic.load(f)

    with io.open(fname, 'r', encoding='cp437') as f:
        contents = f.readlines()
        for l in contents:
            temp = np.zeros(len(word2vec[0]), dtype=float)
            all_words = tokenizer.tokenize(l)
            lexicon = list(all_words)
            lexicon = [(lemmatizer.lemmatize(i)).encode('utf8') for i in lexicon]
            for word in lexicon:
                temp += word2vec[vocab[word]]
            sen2vec.append([temp, output])
    return sen2vec

def get_2D_sen2vec(fname, word2vec_fname, output, m):
    sen2vec = []
    with open(word2vec_fname, "r") as f:
        word2vec, vocab, ivocab = pic.load(f)

    with io.open(fname, 'r', encoding='cp437') as f:
        contents = f.readlines()
        for l in contents:
            temp = np.zeros([m, len(word2vec[0])], dtype=float)
            count = 0
            all_words = tokenizer.tokenize(l)
            lexicon = list(all_words)
            lexicon = [(lemmatizer.lemmatize(i)).encode('utf8') for i in lexicon]
            for word in lexicon:
                temp[count] = word2vec[vocab[word]]
                count += 1
            sen2vec.append([temp, output])
    return sen2vec