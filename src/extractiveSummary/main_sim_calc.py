import csv
import pdb
import json
import ast
import tf_sim
import word_embed_sim
import bert_embed_sim

from Embed_init import Embed_Vocab

if __name__=='__main__':


    annotation_task_name = 'Annotation_Task_Gold_Pilot1'

    alg = 'fasttext'   #{'tf', 'fasttext', 'bert'}
    options = {'embed_dim': 50,      #{50, 300}
               'embed_func':'max',  #{'mean', 'max'}
               'enrich': True
               }

    path_to_hitApp_data = '../../data/extractSum_data/{}.tsv'.format(uhrs_task_name)

    hit_logs = {}
    with open(path_to_hitApp_data, encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            judgement = row

            data_index = judgement['data_index']
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

    if alg == 'tf':
        rank_out_file_name = '../../logs/extractSumRes/{}_rank_{}.txt'.format(alg, uhrs_task_name)
        tf_sim.compute_ranking_tf(hit_logs, rank_out_file_name)
    elif alg == 'fasttext':
        rank_out_file_name = '../../logs/extractSumRes/{}_d{}_{}_rank_{}.txt'.\
            format(alg, options['embed_dim'], options['embed_func'], uhrs_task_name)
        avo_embed_loader = Embed_Vocab(embed_dim=options['embed_dim'])
        embed_vocab = avo_embed_loader.model.wv
        word_embed_sim.compute_ranking_word_embed(hit_logs, rank_out_file_name, options, embed_vocab)
    elif alg == 'bert':
        rank_out_file_name = '../../logs/extractSumRes/{}_rank_{}_{}.txt'.format(alg, options['embed_func'], uhrs_task_name)
        bert_embed_sim.compute_ranking_bert_embed(hit_logs, rank_out_file_name, options)
    else:
        raise NotImplementedError

    print('Done !')
