import numpy as np

def get_wiki_glove_vector(fname, words):
    vectors = {}
    W = []
    vocab = {}
    ivocab = {}
    f1 = open(fname, 'rb').read()
    for line in f1.splitlines():
        temp = line.split()
        vectors[temp[0]] = map(float, temp[1:])

    vocab_size = len(words)

    for i in range(len(words)):
        W.append(vectors[words[i]])
        vocab[words[i]] = i
        ivocab[i] = words[i]
    W = np.array(W)
    # normalize each word vector to unit variance
    #print W[0:2], W[-1:-3]
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    return W_norm, vocab, ivocab

def get_unknown_vocab(fname, words):
    f2 = open(fname, 'r').read()
    vocab = f2.splitlines()
    known_vocab = []
    unknown_vocab = []
    for word in words:
        if word in vocab:
            known_vocab.append(word)
        else:
            unknown_vocab.append(word)
    return known_vocab, unknown_vocab

def get_unknown_vec(words, word2vec, vocab, ivocab):
    old_size = len(vocab)
    word2vec = list(word2vec)
    for i in range(len(words)):
        word2vec.append(np.random.uniform(-0.25, 0.25, len(word2vec[0])))
        vocab[words[i]] = i+old_size
        ivocab[i+old_size] = words[i]
    return word2vec, vocab, ivocab
