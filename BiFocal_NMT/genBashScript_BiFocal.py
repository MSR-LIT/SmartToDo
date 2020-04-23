

device = 1
tokenizer = 'spacy'
start_index = 400
path_to_bash = './run_cuda{}.sh'.format(device)

with open(path_to_bash, 'w') as fptr:

    for index in range(start_index, 512, 2):
        fptr.write('CUDA_VISIBLE_DEVICES={} python train.py --config "BiFocal_config_final/{}/config-{}.yml"\n'
                   .format(device, tokenizer, index))
        fptr.write('CUDA_VISIBLE_DEVICES={} python translate.py -model checkpoints_BiFocal_final/avocado.{}_tokenized/{}/model_step_best.pt'
                   ' -src BiFocal_seq2seq_final_data/avocado.{}_tokenized/src-valid.txt -qry BiFocal_seq2seq_final_data/avocado.{}_tokenized/qry-valid.txt -output logs_BiFocal_final/avocado.{}_tokenized/pred-{}-valid.txt '
                   '-verbose --tgt BiFocal_seq2seq_final_data/avocado.{}_tokenized/tgt-valid.txt --max_length 50 --gpu 0\n'
                   .format(device, tokenizer, index, tokenizer, tokenizer, tokenizer, index, tokenizer))
        fptr.write('CUDA_VISIBLE_DEVICES={} python translate.py -model checkpoints_BiFocal_final/avocado.{}_tokenized/{}/model_step_best.pt'
            ' -src BiFocal_seq2seq_final_data/avocado.{}_tokenized/src-test.txt -qry BiFocal_seq2seq_final_data/avocado.{}_tokenized/qry-test.txt -output logs_BiFocal_final/avocado.{}_tokenized/pred-{}-test.txt '
            '-verbose --tgt BiFocal_seq2seq_final_data/avocado.{}_tokenized/tgt-test.txt --max_length 50 --gpu 0\n'
            .format(device, tokenizer, index, tokenizer, tokenizer, tokenizer, index, tokenizer))
