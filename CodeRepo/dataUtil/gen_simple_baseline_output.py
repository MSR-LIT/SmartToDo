import csv
import pdb
import json
import ast
import re
import numpy as np
import spacy
import string


MAX_TARGET_LEN = 50

nlp = spacy.load("en_core_web_sm")

def get_filtered_tokens_spacy(text):

    doc = nlp(text, disable=["ner", "parser", "tagger"])
    tokenized_text = [str(tok).lower() for tok in doc]
    filtered_token_list = [tok for tok in tokenized_text if re.match('^[a-z]|[?]', tok)]
    return filtered_token_list


def filter_summary_old(summ_list):

    filtered_sum = ''
    min_summ_len = float('inf')
    for summ in summ_list:
        summ = summ.lower().strip()
        summ = ''.join(ch for ch in summ if ch.isalpha() or ch == ' ')
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


    path_to_pred_file = '../data/baseline_final_data/pred-all.txt'

    inp_len_stats = []
    target_len_stats = []

    with open(path_to_pred_file, 'w') as fptr_pred:

        for data_index in hit_logs:
            inp_to = hit_logs[data_index]['current_sent_to'].split(';')[0]
            inp_sub = hit_logs[data_index]['current_subject']
            inp_high = hit_logs[data_index]['highlight']

            gold_target = filter_summary(hit_logs[data_index]['summary'])

            if gold_target == 'none':
                continue

            inp_to_tokens = get_filtered_tokens_spacy(inp_to)
            inp_sub_tokens = get_filtered_tokens_spacy(inp_sub)
            inp_high_tokens = get_filtered_tokens_spacy(inp_high)

            if len(inp_to_tokens) > 0:
                name_tok = inp_to_tokens[0]
                if '@' in name_tok:
                    name_tok = name_tok.split('@')[0]
                    name_tok = name_tok.split('.')[0]
                else:
                    name_tok = inp_to_tokens[0]
                name_tok = [name_tok]
            else:
                name_tok = []


            pred_tokens = name_tok + inp_high_tokens + inp_sub_tokens
            pred_str = ' '.join(pred_tokens)


            print('pred = {} \n tgy = {}\n'
                  .format(pred_str, gold_target))

            fptr_pred.write('{}\n'.format(pred_str))


    pdb.set_trace()
    print('Done !')