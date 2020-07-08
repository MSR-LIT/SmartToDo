import numpy as np
import codecs

token_type = 'spacy'
path_to_all_src = '../data/SmartToDo_seq2seq_data/src-all.txt'.format(token_type)
path_to_all_tgt = '../data/SmartToDo_seq2seq_data/tgt-all.txt'.format(token_type)


TRAIN_SIZE = 7349
VAL_SIZE = 1000
TEST_SIZE = 1000

pair_dic = {}
counter = 0
all_body = open(path_to_all_src,'r').readlines()
all_todo = open(path_to_all_tgt,'r').readlines()
for i,line in enumerate(all_body):
    body = line.rstrip()
    todo = all_todo[i].rstrip()
    pair_dic[counter] = (body, todo)
    counter += 1

np.random.seed(0)
rnd_index = np.random.permutation(counter)
print('Total # Points = {}'.format(counter))

# Shuffle and store train data
path_to_train_src = '../data/SmartToDo_seq2seq_data/src-train.txt'.format(token_type)
path_to_train_tgt= '../data/SmartToDo_seq2seq_data/tgt-train.txt'.format(token_type)
start_flag = True

with codecs.open(path_to_train_src, 'w', 'utf-8') as fptr_src, codecs.open(path_to_train_tgt, 'w', 'utf-8') as fptr_tgt:
    for i in range(0, TRAIN_SIZE):
           if start_flag:
               fptr_src.write('{}'.format(pair_dic[rnd_index[i]][0]))
               fptr_tgt.write('{}'.format(pair_dic[rnd_index[i]][1]))
               start_flag = False
           else:
               fptr_src.write('\n{}'.format(pair_dic[rnd_index[i]][0]))
               fptr_tgt.write('\n{}'.format(pair_dic[rnd_index[i]][1]))


# Store validation data
path_to_valid_src = '../data/SmartToDo_seq2seq_data/src-valid.txt'.format(token_type)
path_to_valid_tgt = '../data/SmartToDo_seq2seq_data/tgt-valid.txt'.format(token_type)
start_flag = True
with codecs.open(path_to_valid_src, 'w', 'utf-8') as fptr_src, codecs.open(path_to_valid_tgt, 'w', 'utf-8') as fptr_tgt:
    for i in range(TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE):
        if start_flag:
            fptr_src.write('{}'.format(pair_dic[rnd_index[i]][0]))
            fptr_tgt.write('{}'.format(pair_dic[rnd_index[i]][1]))
            start_flag = False
        else:
            fptr_src.write('\n{}'.format(pair_dic[rnd_index[i]][0]))
            fptr_tgt.write('\n{}'.format(pair_dic[rnd_index[i]][1]))

# Store Test data
path_to_test_src = '../data/SmartToDo_seq2seq_data/src-test.txt'.format(token_type)
path_to_test_tgt = '../data/SmartToDo_seq2seq_data/tgt-test.txt'.format(token_type)
start_flag = True
with codecs.open(path_to_test_src, 'w', 'utf-8') as fptr_src, codecs.open(path_to_test_tgt, 'w', 'utf-8') as fptr_tgt:
    for i in range(TRAIN_SIZE+VAL_SIZE, counter):
        if start_flag:
            fptr_src.write('{}'.format(pair_dic[rnd_index[i]][0]))
            fptr_tgt.write('{}'.format(pair_dic[rnd_index[i]][1]))
            start_flag = False
        else:
            fptr_src.write('\n{}'.format(pair_dic[rnd_index[i]][0]))
            fptr_tgt.write('\n{}'.format(pair_dic[rnd_index[i]][1]))

print('Split seq2seq data!')
