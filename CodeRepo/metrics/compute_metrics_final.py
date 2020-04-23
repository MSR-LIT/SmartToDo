
import pdb
from metric_util import compute_meteor_rouge

if __name__=='__main__':

        path_to_gold_final =  '../data/Gold_seq2seq_final_data/gold-tgt-test.txt'

        best_rouge_L = -1
        best_index = 0
        tokenizer = 'spacy'

        for index in range(960, 961):

            path_to_valid = '../logs/logs_QuerySent_final/avocado.{}_tokenized/pred-{}-test.txt'.format(tokenizer, index)
            
            #try:
            rouge_L, rouge_1, rouge_2, meteor, bleu4 = compute_meteor_rouge(path_to_gold_final, path_to_valid, tokenizer = tokenizer)
            #except:
            #    rouge_L = 0
             
            pdb.set_trace()
            print('Index = {}, rouge-L = {}, rouge_1 = {}, rouge_2 = {}, blue4 = {}'
                  .format(index, rouge_L, rouge_1, rouge_2, bleu4))

            if rouge_L > best_rouge_L:
                best_rouge_L = rouge_L
                best_index = index


        print('Best rouge-L = {}, Best Index = {}'.format(best_rouge_L, best_index))
