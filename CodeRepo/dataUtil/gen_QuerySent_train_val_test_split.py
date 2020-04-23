
import pdb
import pickle
import numpy as np
import codecs

token_type = 'spacy' #{spacy, bert}
path_to_all_src = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/src-all.txt'.format(token_type)
path_to_all_qry = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/qry-all.txt'.format(token_type)
path_to_all_tgt = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/tgt-all.txt'.format(token_type)


TRAIN_SIZE = 7349
VAL_SIZE = 1000
TEST_SIZE = 1000

tuple_dic = {}
counter = 0
all_scr = open(path_to_all_src,'r').readlines()
all_qry = open(path_to_all_qry,'r').readlines()
all_tgt = open(path_to_all_tgt,'r').readlines()
for i,line in enumerate(all_scr):
    src = line.rstrip()
    qry = all_qry[i].rstrip()
    tgt = all_tgt[i].rstrip()
    tuple_dic[counter] = (src, qry, tgt)
    counter += 1

np.random.seed(0)
rnd_index = np.random.permutation(counter)

# Shuffle and store train data
path_to_train_src = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/src-train.txt'.format(token_type)
path_to_train_qry = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/qry-train.txt'.format(token_type)
path_to_train_tgt = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/tgt-train.txt'.format(token_type)

start_flag = True
with codecs.open(path_to_train_src, 'w', 'utf-8') as fptr_src, \
        codecs.open(path_to_train_qry, 'w', 'utf-8') as fptr_qry, \
        codecs.open(path_to_train_tgt, 'w', 'utf-8') as fptr_tgt:
    for i in range(0, TRAIN_SIZE):
           if start_flag:
               fptr_src.write('{}'.format(tuple_dic[rnd_index[i]][0]))
               fptr_qry.write('{}'.format(tuple_dic[rnd_index[i]][1]))
               fptr_tgt.write('{}'.format(tuple_dic[rnd_index[i]][2]))
               start_flag = False
           else:
               fptr_src.write('\n{}'.format(tuple_dic[rnd_index[i]][0]))
               fptr_qry.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))
               fptr_tgt.write('\n{}'.format(tuple_dic[rnd_index[i]][2]))


# Store validation data
path_to_valid_src = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/src-valid.txt'.format(token_type)
path_to_valid_qry = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/qry-valid.txt'.format(token_type)
path_to_valid_tgt = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/tgt-valid.txt'.format(token_type)

start_flag = True
with codecs.open(path_to_valid_src, 'w', 'utf-8') as fptr_src, \
        codecs.open(path_to_valid_qry, 'w', 'utf-8') as fptr_qry, \
        codecs.open(path_to_valid_tgt, 'w', 'utf-8') as fptr_tgt:
    for i in range(TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE):
        if start_flag:
            fptr_src.write('{}'.format(tuple_dic[rnd_index[i]][0]))
            fptr_qry.write('{}'.format(tuple_dic[rnd_index[i]][1]))
            fptr_tgt.write('{}'.format(tuple_dic[rnd_index[i]][2]))
            start_flag = False
        else:
            fptr_src.write('\n{}'.format(tuple_dic[rnd_index[i]][0]))
            fptr_qry.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))
            fptr_tgt.write('\n{}'.format(tuple_dic[rnd_index[i]][2]))

# Store Test data
path_to_test_src = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/src-test.txt'.format(token_type)
path_to_test_qry = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/qry-test.txt'.format(token_type)
path_to_test_tgt = '../data/QuerySent_seq2seq_final_data/avocado.{}_tokenized/tgt-test.txt'.format(token_type)

start_flag = True
with codecs.open(path_to_test_src, 'w', 'utf-8') as fptr_src, \
        codecs.open(path_to_test_qry, 'w', 'utf-8') as fptr_qry, \
        codecs.open(path_to_test_tgt, 'w', 'utf-8') as fptr_tgt:
    for i in range(TRAIN_SIZE+VAL_SIZE, counter):
        if start_flag:
            fptr_src.write('{}'.format(tuple_dic[rnd_index[i]][0]))
            fptr_qry.write('{}'.format(tuple_dic[rnd_index[i]][1]))
            fptr_tgt.write('{}'.format(tuple_dic[rnd_index[i]][2]))
            start_flag = False
        else:
            fptr_src.write('\n{}'.format(tuple_dic[rnd_index[i]][0]))
            fptr_qry.write('\n{}'.format(tuple_dic[rnd_index[i]][1]))
            fptr_tgt.write('\n{}'.format(tuple_dic[rnd_index[i]][2]))

