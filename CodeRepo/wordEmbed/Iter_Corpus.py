
import pdb

class Tokenize_Sent(object):

    def __init__(self, filename, corpus_size):
        self.filename = filename
        if corpus_size == -1:
            self.size = self.getCorpusSize(filename)
        else:
            self.size = corpus_size

    def getCorpusSize(self, filename):
        with open(self.filename, 'r') as f:
            count = 0
            for line in f:
                 count += 1
            return count

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:

                token_seq = line.rstrip().lower().split(' ')
                yield (token_seq)


def load_Bulk_Corpus(filename):

    with open(filename, 'r') as f:
        sent_list = []
        for line in f:
            token_seq = line.rstrip().lower().split(' ')
            sent_list.append(token_seq)
        return sent_list


if __name__=='__main__':

    corpus_path = '../data/wordEmbed_data/Tokenized_Sentences.txt'
    gen_sent = Tokenize_Sent(corpus_path, corpus_size = -1)

    for item in gen_sent:
        print(item)
        pdb.set_trace()

