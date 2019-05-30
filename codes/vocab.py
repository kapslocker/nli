import bcolz
import numpy as np
import pickle

def build_vocab():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='../data/glove.6B/6B.50d.dat', mode='w')

    with open('../data/glove.6B/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = len(word2idx)
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir='../data/glove.6B/6B.50d.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open('../data/glove.6B/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open('../data/glove.6B/6B.50_idx.pkl', 'wb'))

if __name__ == '__main__':
    build_vocab()
