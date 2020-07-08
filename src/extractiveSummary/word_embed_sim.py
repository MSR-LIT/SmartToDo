

import Util
import numpy as np
import pdb

def compute_ranking_point_word_embed(highlight, subject, sent_list, options, embed_vocab):

   init_corpus = sent_list.copy()
   init_corpus.extend([subject, highlight])
   top_word_list = Util.get_sorted_count_terms(init_corpus, top_k = 10)
   enriched_sent = highlight + ' ' + subject + ' ' + ' '.join(top_word_list)

   embed_dim = options['embed_dim']
   embed_func = options['embed_func']

   #Obtain embedding of enriched_sent
   enriched_vec = Util.get_embedding_sent_fast_text(enriched_sent, embed_dim, embed_func, embed_vocab)
   norm_vec = np.linalg.norm(enriched_vec)
   if norm_vec != 0:
       enriched_vec = enriched_vec/norm_vec


   score = np.zeros(len(sent_list))
   for i in range(len(sent_list)):
       sent_vec = Util.get_embedding_sent_fast_text(sent_list[i], embed_dim, embed_func, embed_vocab)
       norm_vec = np.linalg.norm(sent_vec)
       if norm_vec != 0:
        sent_vec = sent_vec/norm_vec
       score[i] = np.dot(enriched_vec, sent_vec)

   indx = np.flip(np.argsort(score))
   # print('H : {},  Sub : {}'.format(highlight, subject))
   # print('Context : {}'.format(sent_list[indx[0]]))

   return np.around(score[indx], decimals=4), indx


def compute_ranking_word_embed(data, out_fname, options, embed_vocab):

    with open(out_fname, 'w') as fptr:
        for data_index in data:
            highlight = data[data_index]['highlight']
            subject = data[data_index]['current_subject']
            sent_list = data[data_index]['sent-list']

            ranked_score, ranked_list = compute_ranking_point_word_embed(highlight, subject, sent_list, options, embed_vocab)
            assert len(ranked_list) == len(sent_list)

            item_list = []
            for i in range(len(ranked_list)):
                item = str(ranked_list[i])+','+str(ranked_score[i])
                item_list.append(item)
            ranked_str = ';'.join(item_list)

            #pdb.set_trace()

            fptr.write('{}\t{}\n'.format(data_index, ranked_str))
