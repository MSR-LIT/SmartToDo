
import pdb
import pickle
import numpy as np
import codecs

token_type = 'spacy' #{spacy, bert}
path_to_all_body = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/all.body'.format(token_type)
path_to_all_todo = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/all.todo'.format(token_type)


TRAIN_SIZE = 4500
VAL_SIZE = 200
TEST_SIZE = 217

pair_dic = {}
counter = 0
all_body = open(path_to_all_body,'r').readlines()
all_todo = open(path_to_all_todo,'r').readlines()
for i,line in enumerate(all_body):
    body = line.rstrip()
    todo = all_todo[i].rstrip()
    pair_dic[counter] = (body, todo)
    counter += 1

np.random.seed(0)
rnd_index = np.random.permutation(counter)

# Shuffle and store train data
path_to_train_body = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/train.body'.format(token_type)
path_to_train_todo = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/train.todo'.format(token_type)
start_flag = True
with codecs.open(path_to_train_body, 'w', 'utf-8') as fptr_body, codecs.open(path_to_train_todo, 'w', 'utf-8') as fptr_todo:
    for i in range(0, TRAIN_SIZE):
           if start_flag:
               fptr_body.write('{}'.format(pair_dic[rnd_index[i]][0]))
               fptr_todo.write('{}'.format(pair_dic[rnd_index[i]][1]))
               start_flag = False
           else:
               fptr_body.write('\n{}'.format(pair_dic[rnd_index[i]][0]))
               fptr_todo.write('\n{}'.format(pair_dic[rnd_index[i]][1]))


# Store validation data
path_to_valid_body = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/valid.body'.format(token_type)
path_to_valid_todo = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/valid.todo'.format(token_type)
start_flag = True
with codecs.open(path_to_valid_body, 'w', 'utf-8') as fptr_body, codecs.open(path_to_valid_todo, 'w', 'utf-8') as fptr_todo:
    for i in range(TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE):
        if start_flag:
            fptr_body.write('{}'.format(pair_dic[rnd_index[i]][0]))
            fptr_todo.write('{}'.format(pair_dic[rnd_index[i]][1]))
            start_flag = False
        else:
            fptr_body.write('\n{}'.format(pair_dic[rnd_index[i]][0]))
            fptr_todo.write('\n{}'.format(pair_dic[rnd_index[i]][1]))

# Store Test data
path_to_test_body = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/test.body'.format(token_type)
path_to_test_todo = '../data/seq2seq_data/avocado.{}_tokenized.body-todo/test.todo'.format(token_type)
start_flag = True
with codecs.open(path_to_test_body, 'w', 'utf-8') as fptr_body, codecs.open(path_to_test_todo, 'w', 'utf-8') as fptr_todo:
    for i in range(TRAIN_SIZE+VAL_SIZE, counter):
        if start_flag:
            fptr_body.write('{}'.format(pair_dic[rnd_index[i]][0]))
            fptr_todo.write('{}'.format(pair_dic[rnd_index[i]][1]))
            start_flag = False
        else:
            fptr_body.write('\n{}'.format(pair_dic[rnd_index[i]][0]))
            fptr_todo.write('\n{}'.format(pair_dic[rnd_index[i]][1]))

