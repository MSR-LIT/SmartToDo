
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import spacy
import scipy
import numpy as np
np.set_printoptions(precision=4)

nlp = spacy.load('en_core_web_sm')
import Util
import pdb

def compute_ranking_point_tf(highlight, subject, sent_list):

   init_corpus = sent_list.copy()
   init_corpus.extend([subject, highlight])
   top_word_list = Util.get_sorted_count_terms(init_corpus, top_k = 10)
   enriched_sent = highlight + ' ' + subject + ' ' + ' '.join(top_word_list)
   print('Top 10 words : {}'.format(top_word_list))


   new_corpus = sent_list.copy()
   new_corpus.extend([enriched_sent])
   tf_mat = CountVectorizer(preprocessor=Util.custom_pre_processor, tokenizer=Util.spacy_tokenizer, binary=True).fit_transform(new_corpus)

   candidate_enc = tf_mat[0:-1,:]
   candidate_enc = normalize(candidate_enc, norm='l2', axis=1)
   enriched_enc = tf_mat[-1, :].reshape(-1, 1)

   score = candidate_enc.dot(enriched_enc)
   score = scipy.sparse.csr_matrix.toarray(score).flatten()

   indx = np.flip(np.argsort(score))
   print('H : {}'.format(highlight))
   print('Context : {}'.format(new_corpus[indx[0]]))

   return np.around(score[indx], decimals = 4), indx


def compute_ranking_tf(data, out_fname):

    with open(out_fname, 'w') as fptr:
        for data_index in data:
            highlight = data[data_index]['highlight']
            subject = data[data_index]['current_subject']
            sent_list = data[data_index]['sent-list']

            ranked_score, ranked_list = compute_ranking_point_tf(highlight, subject, sent_list)
            assert len(ranked_list) == len(sent_list)

            item_list = []
            for i in range(len(ranked_list)):
                item = str(ranked_list[i])+','+str(ranked_score[i])
                item_list.append(item)
            ranked_str = ';'.join(item_list)

            fptr.write('{}\t{}\n'.format(data_index, ranked_str))



