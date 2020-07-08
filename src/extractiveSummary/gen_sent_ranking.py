import csv
import pdb
import json
import ast

import word_embed_sim
from Embed_init import Embed_Vocab

if __name__=='__main__':

    task_file_name = 'SmartToDo_dataset'

    alg = 'fasttext'
    options = {'embed_dim': 300,      #{50, 300}
                'embed_func':'max',  #{'mean', 'max'}
                'enrich': True
                }

    hit_logs = {}

    path_to_hitApp_data = '../../data/Annotations/{}.tsv'.format(task_file_name)

    with open(path_to_hitApp_data, encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')

        for row in reader:
          judgement = row

          data_index = int(judgement['data_index'])

          sent_dic = json.loads(judgement['sent_json'])
          num_candidates = len(sent_dic)

          if data_index not in hit_logs:
              hit_logs[data_index] = {}

          hit_logs[data_index]['current_subject'] = judgement['current_subject']
          hit_logs[data_index]['highlight'] = judgement['highlight']
          temp = judgement['words_json']
          candidate_list = ast.literal_eval(temp)
          hit_logs[data_index]['sent-list'] = candidate_list

          assert len(candidate_list) == num_candidates, print("Error in Candidate count !")


    if alg == 'fasttext':
        rank_out_file_name = '../../data/Gold_seq2seq_final_data/sent_ranked_{}.txt'.format(alg)
        avo_embed_loader = Embed_Vocab(embed_dim=options['embed_dim'])
        embed_vocab = avo_embed_loader.model.wv
        word_embed_sim.compute_ranking_word_embed(hit_logs, rank_out_file_name, options, embed_vocab)
    else:
        raise NotImplementedError