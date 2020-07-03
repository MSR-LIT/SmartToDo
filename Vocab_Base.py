import pandas as pd
import pickle
import pdb
import time

import encrypt_util

def save_email_ids():

    path_to_merged_UHRS = './data/UHRS_judgements/merged_UHRS.tsv'

    merged_data = pd.read_csv(path_to_merged_UHRS, delimiter='\t')

    emails_ids = merged_data['current_id']
    emails_ids = emails_ids.drop_duplicates()

    path_to_email_ids = './data/email_ids.csv'

    emails_ids.to_csv(path_to_email_ids, index = False)


def gen_vocab(email_ids):

    fwd_vocab = {}
    back_vocab = {}

    fwd_vocab[''] = '$0$'
    fwd_vocab['<No'] = '$1$'
    fwd_vocab['Content>'] = '$2$'
    fwd_vocab['From:'] = '$3$'
    fwd_vocab['To:'] = '$4$'
    fwd_vocab['Cc:'] = '$5$'
    fwd_vocab['application.).'] = '$6$'
    fwd_vocab['Wed..'] = '$7$'
    fwd_vocab['o.k..'] = '$8$'
    fwd_vocab['>From:'] = '$9$'
    fwd_vocab['>To:'] = '$10$'
    fwd_vocab['>Cc:'] = '$11$'
    fwd_vocab['>Date:'] = '$12$'
    fwd_vocab['>Message-ID:'] = '$13$'
    fwd_vocab['>Reply-To:'] = '$14$'
    fwd_vocab['FROM:'] = '$15$'
    fwd_vocab['TO:'] = '$16$'
    fwd_vocab['CC:'] = '$17$'
    fwd_vocab['\n'] = '$18$'
    fwd_vocab['\t'] = '$19$'
    fwd_vocab[';'] = '$20$'
    fwd_vocab['.'] = '$21$'
    fwd_vocab['?)'] = '$22$'
    fwd_vocab['?'] = '$23$'
    fwd_vocab['!'] = '$24$'
    fwd_vocab['\''] = '$25$'
    fwd_vocab[').'] = '$26$'
    fwd_vocab[']'] = '$27$'
    fwd_vocab['pdf>>'] = '$28$'
    fwd_vocab['doc>>'] = '$29$'

    for (key, val) in fwd_vocab.items():
        back_vocab[val] = key

    start = time.time()
    counter = len(fwd_vocab)
    # Extract vocab from email texts.
    print('Extracting vocab from email texts ...')
    for i in range(len(email_ids)):
        if (i+1)%500 == 0:
            print('Processed {}/{}'.format(i+1, len(email_ids)))
        #print('{}, {}'.format(i+1, email_ids.iloc[i, 0]))
        email_id = email_ids.iloc[i, 0]
        email_text = encrypt_util.get_raw_corpus_text(email_id)
        tokens = email_text.split()
        for tok in tokens:
            if tok not in fwd_vocab:
                fwd_vocab[tok] = '$' + str(counter) + '$'
                back_vocab['$' + str(counter) + '$'] = tok
                counter += 1
    print('Done.')

    assert len(fwd_vocab) == len(back_vocab), "ERROR ! Coding dictionaries not of equal length!"
    
    with open('./data/fwd_vocab.pkl', 'wb') as handle:
        pickle.dump(fwd_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./data/back_vocab.pkl', 'wb') as handle:
        pickle.dump(back_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Time elapsed = {} minutes'.format((time.time() - start)/60))


if __name__=='__main__':

    path_to_email_ids = './data/email_ids.csv'
    email_ids = pd.read_csv(path_to_email_ids, header = None)
    gen_vocab(email_ids)

