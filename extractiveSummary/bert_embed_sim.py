
from bert_serving.client import BertClient
import Util
import numpy as np
import pdb

def compute_ranking_point_bert_embed(highlight, subject, sent_list, options):

   init_corpus = sent_list.copy()
   init_corpus.extend([subject, highlight])
   top_word_list = Util.get_sorted_count_terms(init_corpus, top_k = 10)
   enriched_sent = highlight + subject + ' '.join(top_word_list)

   #Obtain embedding of all_sentences
   new_corpus = sent_list.copy()
   new_corpus.extend([enriched_sent])

   with BertClient() as bc:
       embeddings = bc.encode(new_corpus)

   enriched_vec = embeddings[-1, :]
   norm_vec = np.linalg.norm(enriched_vec)
   if norm_vec != 0:
       enriched_vec = enriched_vec/norm_vec


   score = np.zeros(len(sent_list))
   for i in range(len(sent_list)):
       sent_vec = embeddings[i, :]
       norm_vec = np.linalg.norm(sent_vec)
       if norm_vec != 0:
        sent_vec = sent_vec/norm_vec
       score[i] = np.dot(enriched_vec, sent_vec)

   indx = np.flip(np.argsort(score))
   print('H : {}'.format(highlight))
   print('Context : {}'.format(sent_list[indx[0]]))

   return np.around(score[indx], decimals=4), indx


def compute_ranking_bert_embed(data, out_fname, options):

    with open(out_fname, 'w') as fptr:
        for data_index in data:
            highlight = data[data_index]['highlight']
            subject = data[data_index]['current_subject']
            sent_list = data[data_index]['sent-list']

            ranked_score, ranked_list = compute_ranking_point_bert_embed(highlight, subject, sent_list, options)
            assert len(ranked_list) == len(sent_list)

            item_list = []
            for i in range(len(ranked_list)):
                item = str(ranked_list[i])+','+str(ranked_score[i])
                item_list.append(item)
            ranked_str = ';'.join(item_list)

            fptr.write('{}\t{}\n'.format(data_index, ranked_str))