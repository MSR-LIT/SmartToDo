
import numpy as np
import codecs

path_to_all_gold = './data/Gold_SmartToDo_seq2seq_data/gold-tgt-all.txt'

TRAIN_SIZE = 7349
VAL_SIZE = 1000
TEST_SIZE = 1000

tuple_dic = {}
counter = 0
all_tgt = open(path_to_all_gold,'r').readlines()

for i, line in enumerate(all_tgt):
    tgt = line.rstrip()
    tuple_dic[counter] = (None, tgt)
    counter += 1

np.random.seed(0)
rnd_index = np.random.permutation(counter)

# Shuffle and store train data
path_to_gold_train = './data/Gold_SmartToDo_seq2seq_data/gold-tgt-train.txt'
start_flag = True
with codecs.open(path_to_gold_train, 'w', 'utf-8') as fptr_gold:
    for i in range(0, TRAIN_SIZE):
           if start_flag:
               fptr_gold.write('{}'.format(tuple_dic[rnd_index[i]][1]))
               start_flag = False
           else:
               fptr_gold.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))



# Shuffle and store validation data
path_to_gold_valid = './data/Gold_SmartToDo_seq2seq_data/gold-tgt-valid.txt'
start_flag = True
with codecs.open(path_to_gold_valid, 'w', 'utf-8') as fptr_gold:
    for i in range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE):
           if start_flag:
               fptr_gold.write('{}'.format(tuple_dic[rnd_index[i]][1]))
               start_flag = False
           else:
               fptr_gold.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))




# Shuffle and store test data
path_to_gold_test = './data/Gold_SmartToDo_seq2seq_data/gold-tgt-test.txt'
start_flag = True
with codecs.open(path_to_gold_test, 'w', 'utf-8') as fptr_gold:
    for i in range(TRAIN_SIZE + VAL_SIZE, counter):
           if start_flag:
               fptr_gold.write('{}'.format(tuple_dic[rnd_index[i]][1]))
               start_flag = False
           else:
               fptr_gold.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))

print('Split Gold summaries!')