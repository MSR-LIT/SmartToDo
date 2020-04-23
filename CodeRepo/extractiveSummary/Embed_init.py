from gensim.models import FastText
import os

class Embed_Vocab(object):

    def __init__(self, corpus = 'Avocado', corpus_size = -1, embed_type = 'fasttext', embed_dim = 50, window_size = 5, max_iter = 10, path_to_corpus = '', save_flag = False):

        self.corpus = corpus
        self.embed_type = embed_type
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.max_iter = max_iter
        self.path_to_corpus = path_to_corpus
        self.save_flag = save_flag

        self.model = None
        self._load()

    def _load(self):

        checkpoint_dir = '../logs/checkpoint_wordEmbed/{}/corpus800k'.format(self.corpus)
        if not os.path.exists(checkpoint_dir):
            print('Checkpoint Dir Does not Exist !')
        else:

            if self.embed_type == 'fasttext':
                self.model = FastText.load(os.path.join(checkpoint_dir, 'fasttext.{}d.model'.format(self.embed_dim)))
            else:
                raise NotImplementedError