
import sys
sys.path.append('../')

import pdb
from processText import loadUtil
from Embed_Vocab import Embed_Vocab

import numpy as np

def gen_embed_init(vocab, corpus, embed_type, Dim, vocab_size,  token_size, vocab_info):

    embed_obj = Embed_Vocab(corpus=corpus, embed_type=embed_type, embed_dim=Dim)
    embed_obj.load()
    embed_vocab = embed_obj.model.wv

    embed_init = np.zeros((vocab_size, Dim))
    intersect_count = 0
    diff_count = 0
    for word in vocab:
        index = vocab[word]
        if word in embed_vocab:
            vec = embed_vocab[word]
            embed_init[index] = vec
            intersect_count += 1
        else:
            print(word)
            rnd_vec = np.random.uniform(low=-0.5, high=0.5, size=(1, Dim))
            embed_init[index] = rnd_vec
            diff_count += 1

    embed_init[0] = np.zeros((1, Dim))
    print('# Intersect Terms = {}, Diff. Terms = {}'.format(intersect_count, diff_count))

    embed_info = corpus[0:3]+'-'+embed_type
    np.save('../../data/DeepModel_clean_Processed_AllText/embed_init_{}_{}.{}.{}d.npy'.format(vocab_info, embed_info, token_size, Dim), embed_init)

if __name__=='__main__':

    num_th = 10
    remove_names = True
    suffix = 'no_ne' if remove_names else 'ne'

    # Load vocab
    vocab_size = num_th * 1000
    path_to_vocab = '../../data/DeepModel_clean_Processed_AllText/vocab_{}k_{}.txt'.format(num_th, suffix)
    vocab_info = '{}k_{}'.format(num_th, suffix)

    Dim = 100
    token_size = 'null'
    corpus = 'Avocado'
    embed_type = 'fasttext'

    vocab = loadUtil.load_vocab(path_to_vocab)
    print('Loaded vocab !')

    gen_embed_init(vocab, corpus, embed_type, Dim, vocab_size, token_size, vocab_info)
