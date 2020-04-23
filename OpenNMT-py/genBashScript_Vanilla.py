

device = 0
tokenizer = 'spacy'
start_index = 1
path_to_bash = './run_cuda{}.sh'.format(device)

with open(path_to_bash, 'w') as fptr:

    for index in range(start_index, 3):
        fptr.write('CUDA_VISIBLE_DEVICES={} python train.py --config "Orig_config_final/{}/config-vanilla{}.yml"\n'
                   .format(device, tokenizer, index))
        fptr.write('CUDA_VISIBLE_DEVICES={} python translate.py -model checkpoints_Orig_final/avocado.{}_tokenized/vanilla{}/model_step_best.pt'
                   ' -src Orig_seq2seq_final_data/avocado.{}_tokenized/src-valid.txt -output logs_Orig_final/avocado.{}_tokenized/pred-vanilla{}-valid.txt '
                   '-replace_unk -verbose --tgt Orig_seq2seq_final_data/avocado.{}_tokenized/tgt-valid.txt --max_length 50 --gpu 0\n'
                   .format(device, tokenizer, index, tokenizer, tokenizer, index, tokenizer))
        fptr.write('CUDA_VISIBLE_DEVICES={} python translate.py -model checkpoints_Orig_final/avocado.{}_tokenized/vanilla{}/model_step_best.pt'
            ' -src Orig_seq2seq_final_data/avocado.{}_tokenized/src-test.txt -output logs_Orig_final/avocado.{}_tokenized/pred-vanilla{}-test.txt '
            '-replace_unk -verbose --tgt Orig_seq2seq_final_data/avocado.{}_tokenized/tgt-test.txt --max_length 50 --gpu 0\n'
            .format(device, tokenizer, index, tokenizer, tokenizer, index, tokenizer))