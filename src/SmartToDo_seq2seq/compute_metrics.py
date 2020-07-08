import numpy as np
import pdb

import rouge
from nltk.translate.bleu_score import sentence_bleu


def compute_bleu_rouge(path_to_gold, path_to_pred):
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                   max_n=4,
                                   limit_length=True,
                                   length_limit=50,
                                   length_limit_type='words',
                                   alpha=0.5, # Default F1_score
                                   weight_factor=1.2,
                                   stemming=True)

        bleu4_scores = []
        rouge_L = {'f': [], 'p': [], 'r': []}
        rouge_1 = {'f': [], 'p': [], 'r': []}
        rouge_2 = {'f': [], 'p': [], 'r': []}

        counter = 0
        with open(path_to_gold, 'r') as f_gold, open(path_to_pred, 'r') as f_pred:
            for ref, hyp in zip(f_gold, f_pred):
                ref = ref.rstrip()
                hyp = hyp.rstrip()

                ref_toks = ref.split()
                hyp_toks = hyp.split()

                score_single_bleu4 = sentence_bleu([ref_toks], hyp_toks)

                scores_rouge = evaluator.get_scores(hyp, ref)

                for key, val in scores_rouge['rouge-l'].items():
                    rouge_L[key].append(val)

                for key, val in scores_rouge['rouge-1'].items():
                    rouge_1[key].append(val)

                for key, val in scores_rouge['rouge-2'].items():
                    rouge_2[key].append(val)

                bleu4_scores.append(score_single_bleu4)

                counter += 1


        print('\nMean Scores {} => '.format(path_to_pred.split('/')[-1]))
        print('Rouge-L : f = {:.4f}, p = {:.4f}, r = {:4f}'.format(np.mean(rouge_L['f']), np.mean(rouge_L['p']), np.mean(rouge_L['r'])))
        print('Rouge-1 : f = {:.4f}, p = {:.4f}, r = {:4f}'.format(np.mean(rouge_1['f']), np.mean(rouge_1['p']), np.mean(rouge_1['r'])))
        print('Rouge-2 : f = {:.4f}, p = {:.4f}, r = {:4f}'.format(np.mean(rouge_2['f']), np.mean(rouge_2['p']), np.mean(rouge_2['r'])))
        print('BLEU4 = {:4f}'.format(np.mean(bleu4_scores)))



if __name__=='__main__':
        
        pred_path_list_final = ['./logs/pred-101-test.txt']
        path_to_gold_final =  '../../data/Gold_SmartToDo_seq2seq_data/gold-tgt-test.txt'
        for path in pred_path_list_final:
           compute_bleu_rouge(path_to_gold_final, path)



