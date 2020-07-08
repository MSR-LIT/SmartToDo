import pandas as pd
import pickle
import pdb
import time

import encode_util

def save_email_ids():

    path_to_annotations = '../data/Orig_Annotations/SmartToDo_dataset.tsv'

    merged_data = pd.read_csv(path_to_annotations, delimiter='\t')

    emails_ids = merged_data['current_id']
    emails_ids = emails_ids.drop_duplicates()

    path_to_email_ids = '../data/email_ids.csv'

    emails_ids.to_csv(path_to_email_ids, index = False)


def gen_vocab(email_ids):

    fwd_vocab = {}
    back_vocab = {}

    fwd_vocab[''] = '$0$'
    fwd_vocab['<No'] = '$1$'
    fwd_vocab['Content>'] = '$2$'
    fwd_vocab['\n'] = '$3$'

    for (key, val) in fwd_vocab.items():
        back_vocab[val] = key

    start = time.time()
    counter = len(fwd_vocab)
    # Extract vocab from email texts.
    print('Extracting vocab from email texts ...')
    for i in range(len(email_ids)):
        if (i+1)%500 == 0:
            print('Processed {}/{}'.format(i+1, len(email_ids)))
        # print('{}, {}'.format(i+1, email_ids.iloc[i, 0]))
        email_id = email_ids.iloc[i, 0]
        email_text = encode_util.get_raw_corpus_text(email_id)
        tokens = email_text.split()
        for tok in tokens:
            if tok not in fwd_vocab:
                fwd_vocab[tok] = '$' + str(counter) + '$'
                back_vocab['$' + str(counter) + '$'] = tok
                counter += 1
            if tok.lower() not in fwd_vocab:
                new_tok = tok.lower()
                fwd_vocab[new_tok] = '$' + str(counter) + '$'
                back_vocab['$' + str(counter) + '$'] = new_tok
                counter += 1

    print('Done.')

    with open('../data/fwd_vocab.pkl', 'wb') as handle:
        pickle.dump(fwd_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../data/back_vocab.pkl', 'wb') as handle:
        pickle.dump(back_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Time elapsed = {} minutes'.format((time.time() - start)/60))

    assert len(fwd_vocab) == len(back_vocab), "Error! Coding dictionaries not of equal length!" 

    
if __name__=='__main__':

    path_to_email_ids = '../data/email_ids.csv'
    email_ids = pd.read_csv(path_to_email_ids, header = None)
    gen_vocab(email_ids)


