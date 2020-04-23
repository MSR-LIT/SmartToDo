
tokenizer_list = ['spacy', 'bert']
vocab_type_list = ['shared', 'separate']
embed_dim_list = [100, 300]
alg_type = 'Orig'

index = 200

option_list = [
    {'rnn_size': 128, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': False},

    {'rnn_size': 64, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': False},

    {'rnn_size': 256, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': False},

    {'rnn_size': 128, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 128, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': False},

    {'rnn_size': 128, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 64, 'optim': 'adadelta',
     'learning_rate': 0.15, 'position_encoding': False},

    {'rnn_size': 128, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 128, 'optim': 'adadelta',
     'learning_rate': 0.15, 'position_encoding': False},

    {'rnn_size': 128, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 128, 'optim': 'adagrad',
     'learning_rate': 0.01, 'position_encoding': False},

    {'rnn_size': 128, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 128, 'optim': 'adadelta',
     'learning_rate': 1.0, 'position_encoding': False},

     {'rnn_size': 64, 'rnn_type': 'GRU', 'encoder_type': 'brnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': False},

     {'rnn_size': 128, 'rnn_type': 'GRU', 'encoder_type': 'brnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': False},

     {'rnn_size': 128, 'rnn_type': 'GRU', 'encoder_type': 'rnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': False},

    {'rnn_size': 256, 'rnn_type': 'GRU', 'encoder_type': 'rnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': False},

    {'rnn_size': 128, 'rnn_type': 'LSTM', 'encoder_type': 'rnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': True},

    {'rnn_size': 128, 'rnn_type': 'GRU', 'encoder_type': 'rnn', 'batch_size': 64, 'optim': 'adagrad',
     'learning_rate': 0.15, 'position_encoding': True},

]

for vocab_type in vocab_type_list:
    for embed_dim in embed_dim_list:
        for i in range(len(option_list)):
            for tokenizer in tokenizer_list:

                with open('../data/Orig_config_final/{}/config-{}.yml'.format(tokenizer, index), 'w') as fptr:

                    fptr.write('save_checkpoint_steps: {}\n'.format(500))
                    fptr.write('keep_checkpoint: {}\n'.format(6))
                    fptr.write('param_init_glorot: \'true\'\n')
                    fptr.write('seed: 0\n')
                    fptr.write('early_stopping: {}\n'.format(5))
                    fptr.write('report_every: {}\n'.format(500))
                    fptr.write('valid_steps: {}\n'.format(500))
                    fptr.write('train_steps: {}\n'.format(100000))

                    fptr.write('\n')
                    fptr.write('copy_attn: \'true\'\n')
                    fptr.write('copy_attn_force: \'true\'\n')
                    fptr.write('dropout: {}\n'.format(0.5))
                    fptr.write('attention_dropout: {}\n'.format(0.5))

                    fptr.write('\n')
                    fptr.write('world_size: {}\n'.format(1))
                    fptr.write('gpu_ranks:\n')
                    fptr.write('- 0\n')

                    fptr.write('\n')

                    data = 'processed_final/avocado.{}_tokenized.{}.copy'.format(tokenizer, vocab_type)
                    save_model = 'checkpoints_Orig_final/avocado.{}_tokenized/{}/model'.format(tokenizer, index)
                    pre_word_vecs_enc = 'processed_final/avocado.{}_tokenized.{}.copy.embeddings.d{}.enc.pt'\
                        .format(tokenizer, vocab_type, embed_dim)
                    pre_word_vecs_dec = 'processed_final/avocado.{}_tokenized.{}.copy.embeddings.d{}.dec.pt' \
                        .format(tokenizer, vocab_type, embed_dim)

                    fptr.write('data: {}\n'.format(data))
                    fptr.write('save_model: {}\n'.format(save_model))
                    fptr.write('pre_word_vecs_enc: {}\n'.format(pre_word_vecs_enc))
                    fptr.write('pre_word_vecs_dec: {}\n'.format(pre_word_vecs_dec))
                    fptr.write('word_vec_size: {}\n'.format(embed_dim))
                    fptr.write('rnn_size: {}\n'.format(option_list[i]['rnn_size']))
                    fptr.write('layers: {}\n'.format(1))
                    fptr.write('rnn_type: {}\n'.format(option_list[i]['rnn_type']))
                    fptr.write('encoder_type: {}\n'.format(option_list[i]['encoder_type']))

                    fptr.write('\n')
                    fptr.write('batch_size: {}\n'.format(option_list[i]['batch_size']))
                    fptr.write('optim: {}\n'.format(option_list[i]['optim']))
                    fptr.write('learning_rate: {}\n'.format(option_list[i]['learning_rate']))
                    fptr.write('adagrad_accumulator_init: {}\n'.format(0.1))
                    fptr.write('max_grad_norm: {}\n'.format(2.0))

                    if option_list[i]['position_encoding'] == True:
                        fptr.write('position_encoding: \'true\'\n')


                index += 1

