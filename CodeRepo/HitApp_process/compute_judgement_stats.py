import csv
import pdb
import json
import ast
import re
import string
import numpy as np

import rouge
from nltk.translate.bleu_score import sentence_bleu

MAX_TARGET_LEN = 50


def filter_summary(summ_list):

    filtered_sum = []

    for summ in summ_list:
        summ = summ.lower().strip('. ')  # Remove period at the end and blank spaces
        summ = summ.translate({ord(c): ' ' for c in string.punctuation}) # Replace punctuations, special chars with space.
        summ = ' '.join(summ.split())  # Remove double spaces
        summ_tok = [tok for tok in summ.split() if re.match('^[0-9]', tok) is None ]  # Remove tokens that start with digits.
        summ = ' '.join(summ_tok)

        filtered_sum.append(summ)

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
                    hit_logs[data_index]['subject-useful'] = []

                hit_logs[data_index]['reply_to_body'] = judgement['reply_to_body']
                hit_logs[data_index]['reply_to_subject'] = judgement['reply_to_subject']
                hit_logs[data_index]['current_subject'] = judgement['current_subject']
                hit_logs[data_index]['current_sent_to'] = judgement['current_sent_to']
                hit_logs[data_index]['highlight'] = judgement['highlight']
                temp = judgement['words_json']
                candidate_list = ast.literal_eval(temp)
                hit_logs[data_index]['sent-list'] = candidate_list
                hit_logs[data_index]['summary'].append(judgement['to_do_summary'])
                hit_logs[data_index]['subject-useful'].append(judgement['subject-useful'])

                assert len(candidate_list) == num_candidates, print("Error in Candidate count !")

    target_len_stats = []
    subject_useful = 0.0

    no_subject = 0.0
    subject_labels_annotator1 = []
    subject_labels_annotator2 = []

    digress_match = 0.0
    digress_count = 0.0

    no_past = 0.0
    
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=50,
                            length_limit_type='words',
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    meteor_scores = []
    bleu4_scores = []
    rouge_L = {'f': [], 'p': [], 'r': []}
    rouge_1 = {'f': [], 'p': [], 'r': []}
    rouge_2 = {'f': [], 'p': [], 'r': []}


    for data_index in hit_logs:
        inp_to = hit_logs[data_index]['current_sent_to'].split(';')[0]
        inp_sub = hit_logs[data_index]['current_subject']
        inp_high = hit_logs[data_index]['highlight']

        gold_target = filter_summary(hit_logs[data_index]['summary'])

        if gold_target[0] == 'none' or gold_target[1] == 'none':
            continue

        if hit_logs[data_index]['subject-useful'][0] == 'Yes' and hit_logs[data_index]['subject-useful'][1] == 'Yes':
            subject_useful += 1

        curr_sub = hit_logs[data_index]['current_subject']
        past_sub = hit_logs[data_index]['reply_to_subject']
        if curr_sub == 'NULL' or curr_sub == '(no subject)':
            no_subject += 1

        if past_sub != '' and past_sub !='NULL' and past_sub !='(no subject)':      
             digress_count += 1
             if past_sub == curr_sub:
                 digress_match += 1
      
        past_body = hit_logs[data_index]['reply_to_body']
        if past_body == '<No Previous Email Content>':
              no_past += 1

        subject_labels_annotator1.append(0 if hit_logs[data_index]['subject-useful'][0] == 'No' else 1)
        subject_labels_annotator2.append(0 if hit_logs[data_index]['subject-useful'][1] == 'No' else 1)
       
        # Compute metrics
        ref = gold_target[0]
        hyp = gold_target[1]

        ref_toks = ref.split()
        hyp_toks = hyp.split()
        target_len_stats.append(min(len(ref_toks), len(hyp_toks)))

        score_single_bleu4 = sentence_bleu([ref_toks], hyp_toks)

        scores_rouge = evaluator.get_scores(hyp, ref)

        for key, val in scores_rouge['rouge-l'].items():
            rouge_L[key].append(val)

        for key, val in scores_rouge['rouge-1'].items():
            rouge_1[key].append(val)

        for key, val in scores_rouge['rouge-2'].items():
            rouge_2[key].append(val)

        bleu4_scores.append(score_single_bleu4)

    print('Pairwise Judgement Scores ==> ')
    print('Rouge-L = {:.2f}, Rouge_1 = {:.2f}, Rouge_2 = {:.2f}, BLEU4 = {:.2f}'
          .format(np.mean(rouge_L['f']), np.mean(rouge_1['f']), np.mean(rouge_2['f']), np.mean(bleu4_scores)))

    print('Mean Target Len = {:.2f}, Median Target Len = {:.2f}'
          .format(np.mean(target_len_stats), np.median(target_len_stats)))


    print('Perc. subject useful = {:.4f}'.format(subject_useful/len(bleu4_scores)))
    pdb.set_trace()
    print('Done !')
