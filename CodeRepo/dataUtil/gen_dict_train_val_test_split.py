
import pdb
import pickle
import numpy as np

path_to_all_pairs = '../data/seq2seq_data/all_pairs.txt'
pair_dic = {}
counter = 0

TRAIN_SIZE = 4500
VAL_SIZE = 200
TEST_SIZE = 217


with open(path_to_all_pairs, 'r') as fptr_all:
    for line in fptr_all.readlines():
           line = line.rstrip().split('\t')
           src_tokens = line[0].split(' ')
           src_len = len(src_tokens)
           trg_tokens = line[1].split(' ')
           pair_dic[counter] = (src_tokens, src_len, trg_tokens)
           counter += 1

rnd_index = np.random.permutation(counter)

# Shuffle and store train data
path_to_train_pairs = '../data/seq2seq_data/train_pairs.pkl'
fptr_train = open(path_to_train_pairs, 'wb')
train_dic = {}
for i in range(0, TRAIN_SIZE):
       train_dic[i-0] = pair_dic[rnd_index[i]]

print(len(train_dic))
pickle.dump(train_dic, fptr_train)

# Store validation data
path_to_val_pairs = '../data/seq2seq_data/val_pairs.pkl'
fptr_val = open(path_to_val_pairs, 'wb')
val_dic = {}
for i in range(TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE):
      val_dic[i-TRAIN_SIZE] = pair_dic[rnd_index[i]]

print(len(val_dic))
pickle.dump(val_dic, fptr_val)

# Store Test data
path_to_test_pairs = '../data/seq2seq_data/test_pairs.pkl'
fptr_test = open(path_to_test_pairs, 'wb')
test_dic = {}
for i in range(TRAIN_SIZE+VAL_SIZE, counter):
       test_dic[i-TRAIN_SIZE-VAL_SIZE] = pair_dic[rnd_index[i]]

print(len(test_dic))
pickle.dump(test_dic, fptr_test)

pdb.set_trace()
print('Done !')

