import csv
import pdb
import json
import ast
import re
import numpy as np
import spacy
import string

MAX_USEFUL_LEN = 100
MAX_TARGET_LEN = 50

nlp = spacy.load("en_core_web_sm")

from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len = 512)

from extractiveSummary import Util

def get_filtered_tokens_bert(text):

    tokenized_text = tokenizer.tokenize(text)
    filtered_token_list = [tok for tok in tokenized_text if re.match('^##[a-z]|^[a-z]|[?]', tok)]
    return filtered_token_list


def get_filtered_tokens_spacy(text):

    doc = nlp(text, disable=["ner", "parser", "tagger"])
    tokenized_text = [str(tok).lower() for tok in doc]
    filtered_token_list = [tok for tok in tokenized_text if re.match('^[a-z]|[?]', tok)]
    return filtered_token_list


def get_ranked_output(path_to_file):

    alg_rank_dic = {}
    with open(path_to_file, 'r') as fptr:
        for line in fptr.readlines():
            line = line.rstrip().split('\t')
            data_index = int(line[0])
            ranked_tuple = line[1].split(';')
            ranked_list = []
            for item in ranked_tuple:
                item = item.split(',')
                if float(item[1]) == 0:
                    continue
                ranked_list.append(int(item[0]))
            alg_rank_dic[data_index] = ranked_list

    return alg_rank_dic


def filter_summary_old(summ_list, token_type):

    filtered_sum = ''
    min_summ_len = float('inf')
    for summ in summ_list:
        summ = summ.lower().strip('. ')
        if token_type == 'bert':
            summ = ''.join(ch for ch in summ if ch.isalpha() or ch == ' ' or ch =='\'')
        summ_len = len(summ.split(' '))
        if summ_len < min_summ_len:
            min_summ_len = summ_len
            filtered_sum = summ

    return filtered_sum


def filter_summary(summ_list):

    filtered_sum = ''
    min_summ_len = float('inf')
    for summ in summ_list:
        summ = summ.lower().strip('. ')  # Remove period at the end and blank spaces
        summ = summ.translate({ord(c): ' ' for c in string.punctuation}) # Replace punctuations, special chars with space.
        summ = ' '.join(summ.split())  # Remove double spaces
        summ_tok = [tok for tok in summ.split() if re.match('^[0-9]', tok) is None ]  # Remove tokens that start with digits.
        summ = ' '.join(summ_tok)
        summ_len = len(summ_tok)
        if summ_len < min_summ_len:
            min_summ_len = summ_len
            filtered_sum = summ

    return filtered_sum



if __name__=='__main__':

    task_file_list = ['UHRS_Task_Pilot2-08-02', 'UHRS_Task_Pilot3-08-09',
                      'UHRS_Task_Augment1_08_12', 'UHRS_Task_Augment2_08_16',
                      'UHRS_Task_Augment3_08_20', 'UHRS_Task_Augment4_08_26', 'UHRS_Task_Augment5_08_30']

    augment_offset = 2520
    augment_flag = 0

    alg = 'fasttext'
    token_type = 'spacy'  #{'bert', 'spacy'}

    hit_logs = {}
    for task_name in task_file_list:
          if 'Augment' in task_name:
              augment_flag = 1
          else:
              augment_flag = 0

          path_to_hitApp_data = '../logs/UHRS_judgements/{}.tsv'.format(task_name)

          with open(path_to_hitApp_data, encoding='utf-8') as tsvfile:
              reader = csv.DictReader(tsvfile, delimiter='\t')
              for row in reader:
                  judgement = row

                  data_index = int(judgement['data_index'])

                  if augment_flag == 1:
                      data_index += augment_offset

                  sent_dic = json.loads(judgement['sent_json'])
                  num_candidates = len(sent_dic)

                  if data_index not in hit_logs:
                      hit_logs[data_index] = {}
                      hit_logs[data_index]['summary'] = []

                  hit_logs[data_index]['current_subject'] = judgement['current_subject']
                  hit_logs[data_index]['current_sent_to'] = judgement['current_sent_to']
                  hit_logs[data_index]['highlight'] = judgement['highlight']
                  temp = judgement['words_json']
                  candidate_list = ast.literal_eval(temp)
                  hit_logs[data_index]['sent-list'] = candidate_list
                  hit_logs[data_index]['summary'].append(judgement['to_do_summary'])

                  assert len(candidate_list) == num_candidates, print("Error in Candidate count !")


    src_out_name = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/src-all.txt'.format(token_type)
    qry_out_name = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/qry-all.txt'.format(token_type)
    tgt_out_name = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/tgt-all.txt'.format(token_type)

    path_to_ranked_sent = '../data/Gold_seq2seq_final_data/sent_ranked_{}.txt'.format(alg)
    ranked_sent_dic = get_ranked_output(path_to_ranked_sent)
    max_K = 1  # Choose maximum of K useful sentences

    src_len_stats = []
    qry_len_stats = []
    tgt_len_stats = []

    start_flag = True

    with open(src_out_name, 'w') as fptr_src, open(qry_out_name, 'w') as fptr_qry, open(tgt_out_name, 'w') as fptr_tgt:
        for data_index in hit_logs:
            inp_to = hit_logs[data_index]['current_sent_to'].split(';')[0]
            inp_sub = hit_logs[data_index]['current_subject']
            inp_high = hit_logs[data_index]['highlight']

            sent_list = hit_logs[data_index]['sent-list']
            ranked_list = ranked_sent_dic[data_index]
            useful_index = ranked_list[0:max_K]
            useful_str = ' '.join(sent_list[index] for index in useful_index)

            gold_target = filter_summary(hit_logs[data_index]['summary'])

            if gold_target == 'none':
                continue

            if token_type == 'bert':
                token_func = get_filtered_tokens_bert
            elif token_type == 'spacy':
                token_func = get_filtered_tokens_spacy
            else:
                raise NotImplementedError

            inp_to_tokens = token_func(inp_to)

            if token_type == 'spacy' and len(inp_to_tokens) > 0:
                name_tok = inp_to_tokens[0]
                if '@' in name_tok:
                    name_tok = name_tok.split('@')[0]
                    name_tok = name_tok.split('.')[0]
                    inp_to_tokens = [name_tok] + inp_to_tokens[1:]

            inp_sub_tokens = token_func(inp_sub)
            inp_high_tokens = token_func(inp_high)
            inp_useful_tokens = token_func(useful_str)
            inp_useful_tokens = inp_useful_tokens[0:MAX_USEFUL_LEN]

            inp_to_tokens = ['<to_s>'] + inp_to_tokens
            # inp_to_tokens = [tok+'|'+'<to>' for tok in inp_to_tokens]
            inp_sub_tokens = ['<sub_s>'] + inp_sub_tokens
            # inp_sub_tokens = [tok+'|'+'<sub>' for tok in inp_sub_tokens]
            inp_useful_tokens = ['<body_s>'] + inp_useful_tokens
            # inp_useful_tokens = [tok+'|'+'<body>' for tok in inp_useful_tokens]


            target_tokens = token_func(gold_target)
            target_tokens = target_tokens[0:MAX_TARGET_LEN]


            src_str = ' '.join(inp_high_tokens)
            qry_str = ' '.join(inp_to_tokens + inp_sub_tokens + inp_useful_tokens)
            tgt_str = ' '.join(target_tokens)


            print('Data Index = {} : Src len = {}, Qry len = {}, Tgt len = {}'
                  .format(data_index, len(inp_high_tokens), len(inp_useful_tokens), len(target_tokens)))

            src_len_stats.append(len(inp_high_tokens))
            qry_len_stats.append(len(inp_useful_tokens))
            tgt_len_stats.append(len(target_tokens))

            print(src_str)
            print(qry_str)
            print(tgt_str)


            if start_flag:
                fptr_src.write('{}'.format(src_str))
                fptr_qry.write('{}'.format(qry_str))
                fptr_tgt.write('{}'.format(tgt_str))
                start_flag = False

            else:
                fptr_src.write('\n{}'.format(src_str))
                fptr_qry.write('\n{}'.format(qry_str))
                fptr_tgt.write('\n{}'.format(tgt_str))
