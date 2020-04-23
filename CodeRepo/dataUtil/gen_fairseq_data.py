import csv
import pdb
import json
import ast
import re
import numpy as np
import spacy

MAX_USEFUL_LEN = 100
MAX_TARGET_LEN = 50

nlp = spacy.load("en_core_web_sm")

from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len = 512)
tokenizer.vocab['<sos>'] = 1
tokenizer.vocab['<eos>'] = 2
tokenizer.vocab['<to>'] = 3
tokenizer.vocab['<sub>'] = 4
tokenizer.vocab['<high>'] = 5
tokenizer.vocab['<sent>'] = 6


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


def filter_summary(summ_list):

    filtered_sum = ''
    min_summ_len = float('inf')
    for summ in summ_list:
        summ = summ.lower().strip('. ')
        # summ = ''.join(ch for ch in summ if ch.isalpha() or ch == ' ')
        summ_len = len(summ.split(' '))
        if summ_len < min_summ_len:
            min_summ_len = summ_len
            filtered_sum = summ

    return filtered_sum



if __name__=='__main__':

    task_file_list = ['UHRS_Task_Pilot2-08-02', 'UHRS_Task_Pilot3-08-09',
                      'UHRS_Task_Augment1_08_12', 'UHRS_Task_Augment2_08_16']

    augment_offset = 2520
    augment_flag = 0

    alg = 'fasttext'
    token_type = 'bert'  #{'bert', 'spacy'}

    hit_logs = {}
    for task_name in task_file_list:
          if 'Augment' in task_name:
              augment_flag = 1
          else:
              augment_flag = 0

          path_to_hitApp_data = '../data/seq2seq_data/{}.tsv'.format(task_name)

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


    path_to_ranked_sent = '../data/seq2seq_data/sent_ranked_fasttext.txt'
    ranked_sent_dic = get_ranked_output(path_to_ranked_sent)

    body_out_name = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/all.body'.format(token_type)
    todo_out_name = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/all.todo'.format(token_type)
    max_K = 1   # Choose maximum of K useful sentences

    inp_len_stats = []
    target_len_stats = []

    start_flag = True

    with open(body_out_name, 'w') as fptr_body, open(todo_out_name, 'w') as fptr_todo:
        for data_index in hit_logs:
            inp_to = hit_logs[data_index]['current_sent_to'].split(';')[0]
            inp_sub = hit_logs[data_index]['current_subject']
            inp_high = hit_logs[data_index]['highlight']

            sent_list = hit_logs[data_index]['sent-list']
            ranked_list = ranked_sent_dic[data_index]
            useful_index = ranked_list[0:max_K]
            useful_str = ' '.join(sent_list[index] for index in useful_index)

            target = filter_summary(hit_logs[data_index]['summary'])

            if target == 'none':
                continue

            if token_type == 'bert':
                token_func = get_filtered_tokens_bert
            elif token_type == 'spacy':
                token_func = get_filtered_tokens_spacy
            else:
                raise NotImplementedError

            inp_to_tokens = token_func(inp_to)
            inp_sub_tokens = token_func(inp_sub)
            inp_high_tokens = token_func(inp_high)
            inp_useful_tokens = token_func(useful_str)

            inp_useful_tokens = inp_useful_tokens[0:MAX_USEFUL_LEN]

            inp_tokens = ['<to>']+inp_to_tokens+['<sub>']+inp_sub_tokens+['<high>']+inp_high_tokens\
                         +['<sent>']+inp_useful_tokens
            inp_str = ' '.join(inp_tokens)

            target_tokens  = token_func(target)
            target_tokens = target_tokens[0:MAX_TARGET_LEN]

            target_len_stats.append(len(target_tokens))
            inp_len_stats.append(len(inp_tokens))


            print('Data Index = {} : Input token len = {}, target token length = {}'
                  .format(data_index, len(inp_tokens), len(target_tokens)))
            target_str = ' '.join(target_tokens)

            print(inp_str)
            print(target_str)


            if start_flag:
                fptr_body.write('{}'.format(inp_str))
                fptr_todo.write('{}'.format(target_str))
                start_flag = False

            else:
                fptr_body.write('\n{}'.format(inp_str))
                fptr_todo.write('\n{}'.format(target_str))


    pdb.set_trace()
    print('Done !')