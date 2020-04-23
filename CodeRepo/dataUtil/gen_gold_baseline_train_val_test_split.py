

import pdb
import pickle
import numpy as np
import codecs

path_to_all_pred = '../data/baseline_final_data/pred-all.txt'
path_to_all_gold = '../data/Gold_seq2seq_final_data/gold-tgt-all.txt'

TRAIN_SIZE = 7349
VAL_SIZE = 1000
TEST_SIZE = 1000

tuple_dic = {}
counter = 0
all_pred = open(path_to_all_pred,'r').readlines()
all_tgt = open(path_to_all_gold,'r').readlines()

for i,line in enumerate(all_pred):
    pred = line.rstrip()
    tgt = all_tgt[i].rstrip()
    tuple_dic[counter] = (pred, tgt)
    counter += 1

np.random.seed(0)
rnd_index = np.random.permutation(counter)

# Shuffle and store train data
path_to_pred_train = '../data/baseline_final_data/pred-train.txt'
path_to_gold_train = '../data/Gold_seq2seq_final_data/gold-tgt-train.txt'
start_flag = True
with codecs.open(path_to_pred_train, 'w', 'utf-8') as fptr_pred, \
        codecs.open(path_to_gold_train, 'w', 'utf-8') as fptr_gold:
    for i in range(0, TRAIN_SIZE):
           if start_flag:
               fptr_pred.write('{}'.format(tuple_dic[rnd_index[i]][0]))
               fptr_gold.write('{}'.format(tuple_dic[rnd_index[i]][1]))
               start_flag = False
           else:
               fptr_pred.write('\n{}'.format(tuple_dic[rnd_index[i]][0]))
               fptr_gold.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))



# Shuffle and store validation data
path_to_pred_valid = '../data/baseline_final_data/pred-valid.txt'
path_to_gold_valid = '../data/Gold_seq2seq_final_data/gold-tgt-valid.txt'
start_flag = True
with codecs.open(path_to_pred_valid, 'w', 'utf-8') as fptr_pred, \
        codecs.open(path_to_gold_valid, 'w', 'utf-8') as fptr_gold:
    for i in range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE):
           if start_flag:
               fptr_pred.write('{}'.format(tuple_dic[rnd_index[i]][0]))
               fptr_gold.write('{}'.format(tuple_dic[rnd_index[i]][1]))
               start_flag = False
           else:
               fptr_pred.write('\n{}'.format(tuple_dic[rnd_index[i]][0]))
               fptr_gold.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))




# Shuffle and store test data
path_to_pred_test = '../data/baseline_final_data/pred-test.txt'
path_to_gold_test = '../data/Gold_seq2seq_final_data/gold-tgt-test.txt'
start_flag = True
with codecs.open(path_to_pred_test, 'w', 'utf-8') as fptr_pred, \
        codecs.open(path_to_gold_test, 'w', 'utf-8') as fptr_gold:
    for i in range(TRAIN_SIZE + VAL_SIZE, counter):
           if start_flag:
               fptr_pred.write('{}'.format(tuple_dic[rnd_index[i]][0]))
               fptr_gold.write('{}'.format(tuple_dic[rnd_index[i]][1]))
               start_flag = False
           else:
               fptr_pred.write('\n{}'.format(tuple_dic[rnd_index[i]][0]))
               fptr_gold.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))
