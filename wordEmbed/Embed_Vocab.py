

from Iter_Corpus import Tokenize_Sent
from gensim.models import FastText
import os
import pdb

class Embed_Vocab(object):

    def __init__(self, corpus = 'Avocado', corpus_size = -1, embed_type = 'word2vec', embed_dim = 50, window_size = 5, max_iter = 10, path_to_corpus = '', save_flag = True):

        self.corpus = corpus
        self.embed_type = embed_type
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.max_iter = max_iter
        self.path_to_corpus = path_to_corpus
        self.save_flag = save_flag

        self.model = None

    def train(self):

        self.gen = Tokenize_Sent(self.path_to_corpus, corpus_size)  #Iterator to read files.
        if self.embed_type == 'fasttext':
            print('Training fasttext model ...')
            # self.model = FastText(sentences = self.gen, size = self.embed_dim, iter =self.max_iter, window = self.window_size, min_count = 5, workers = 1, sg = 1)
            self.model = FastText(size = self.embed_dim, window=self.window_size, min_count=5, workers=4, sg=1)
            self.model.build_vocab(sentences=self.gen)
            self.model.train(sentences=self.gen, total_examples=self.gen.size, epochs=self.max_iter)

            wv = self.model.wv

            print('Words most similar to \'manager\':')
            print(wv.most_similar('manager'))


        else:
            raise NotImplementedError

        if self.save_flag:
            self.save(self.model)

    def save(self, model):

        checkpoint_dir = '../logs/checkpoint_wordEmbed/{}'.format(self.corpus)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model.save(os.path.join(checkpoint_dir, '{}.{}d.model'.format(self.embed_type, self.embed_dim)))


    def load(self):

        checkpoint_dir = '../logs/checkpoint_wordEmbed/{}/corpus800k'.format(self.corpus)
        if not os.path.exists(checkpoint_dir):
            print('Checkpoint Dir Does not Exist !')
        else:

            if self.embed_type == 'fasttext':
                self.model = FastText.load(os.path.join(checkpoint_dir, 'fasttext.{}d.model'.format(self.embed_dim)))
            else:
                raise NotImplementedError

    def _infer(self):

        self.load()
        wv = self.model.wv
        print('hwty' in wv)
        print('##ed' in wv)
        print('##y' in wv)
        print('##mi' in wv)
        print('##ne' in wv)
        print('Vector embedding for \'hello\':')
        print(wv['hello'])

        word_list = ['thanks', 'dear', 'happy', 'sad', 'cost', 'will', 'engine', 'call', 'mail', 'server', 'bug', 'posted', 'inform', 'done',
 'send', 'forward', 'talk', 'update', 'regards', 'best', 'worst', 'http']
        
        for word in word_list:
             print('Words most similar to \'{}\':'.format(word))
             print(wv.most_similar(word))

        
if __name__=='__main__':

    corpus_path = '../data/wordEmbed_data/Tokenized_Sentences.txt'
    corpus = 'Avocado'
    embed_type = 'fasttext'
    embed_dim = 300
    window_size = 5
    max_iter = 10
    corpus_size = -1
    train_flag = False

    embed_obj = Embed_Vocab(corpus = corpus, corpus_size = corpus_size,
                            embed_type =embed_type, embed_dim = embed_dim,
                            window_size = window_size, max_iter = max_iter, path_to_corpus= corpus_path)

    if train_flag:
        embed_obj.train()
    else:
        embed_obj._infer()



