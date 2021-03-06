import csv

import pdb
import token_span_util
import encode_util
import pickle

def gen_encode_token_span_dataset(dev = False):

    task_inp_file_name = 'SmartToDo_dataset'
    task_out_file_name = 'Spans_ToDo_dataset'

    span_logs = {}

    path_to_inp = '../data/Orig_Annotations/{}.tsv'.format(task_inp_file_name)
    path_to_out = '../data/Coded_Annotations/{}.tsv'.format(task_out_file_name)

    fieldnames = ['UniqueID', 'data_index', 'current_id', 'highlight_start', 'highlight_end', 'to_do_summary']

    PATH_TO_FWD_VOCAB = '../data/fwd_vocab.pkl'
    unk_set = set()

    with open(PATH_TO_FWD_VOCAB, 'rb') as handle:
        fwd_vocab = pickle.load(handle)

    print('Encoding data ...')
    with open(path_to_out, 'w', encoding='utf-8', newline='') as out_tsv_file, open(path_to_inp, 'r', encoding='utf-8') as inp_tsv_file:

        writer = csv.DictWriter(out_tsv_file, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        reader = csv.DictReader(inp_tsv_file, delimiter='\t')
        row_count = 0

        for row in reader:

            val_dic = {}
            judgement = row

            val_dic['UniqueID'] = judgement['UniqueID']
            val_dic['data_index'] = judgement['data_index']
            val_dic['current_id'] = judgement['current_id']
            data_index = int(judgement['data_index'])
            highlight = judgement['highlight']
            email_id = judgement['current_id']
            summary = judgement['to_do_summary']

            if data_index not in span_logs:
                start_span, end_span = token_span_util.get_highlight_span(email_id, highlight)
                if start_span == None or end_span == None:
                    continue

                span_logs[data_index] = (start_span, end_span)
                row_count += 1
                if row_count % 1000 == 0:
                    print('Processed {}'.format(row_count))

            val_dic['highlight_start'] = span_logs[data_index][0]
            val_dic['highlight_end'] = span_logs[data_index][1]

            if dev:
                encoded_str = encode_util.encode_sent_dev(summary, fwd_vocab)
                val_dic['to_do_summary'] = encoded_str
                for tok in encoded_str.split():
                    try:
                        if tok[0] != '$' or tok[-1] != '$':
                           unk_set.add(tok)
                    except:
                        pdb.set_trace()

            else:
                val_dic['to_do_summary'] = encode_util.encode_sent(summary, fwd_vocab)

            writer.writerow(val_dic)

    if dev:
        with open('../data/spell_correct_unk_set.txt', 'w') as funk:
            for item in unk_set:
                funk.write('{}\n'.format(item))


def gen_decode_token_span_dataset():

    task_file_name = 'Spans_ToDo_dataset'

    span_logs = {}

    path_to_inp = '../data/Coded_Annotations/{}.tsv'.format(task_file_name)
    path_to_out = '../data/Annotations/{}.tsv'.format(task_file_name)


    row_count = 0
    err_count = 0

    fieldnames = ['UniqueID', 'data_index', 'current_id', 'highlight', 'to_do_summary']

    PATH_TO_BACK_VOCAB = '../data/back_vocab.pkl'

    print('Decoding data ...')
    with open(PATH_TO_BACK_VOCAB, 'rb') as handle:
        back_vocab = pickle.load(handle)

    with open(path_to_out, 'w', encoding='utf-8', newline='') as out_tsv_file, open(path_to_inp, 'r', encoding='utf-8') as inp_tsv_file:

        writer = csv.DictWriter(out_tsv_file, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        reader = csv.DictReader(inp_tsv_file, delimiter='\t')

        for row in reader:

            val_dic = {}
            judgement = row

            val_dic['UniqueID'] = judgement['UniqueID']
            val_dic['data_index'] = judgement['data_index']
            val_dic['current_id'] = judgement['current_id']
            data_index = int(judgement['data_index'])
            email_id = judgement['current_id']
            summary = judgement['to_do_summary']

            try:
                start_span = int(judgement['highlight_start'])
                end_span = int(judgement['highlight_end'])

                row_count += 1
                if row_count % 1000 == 0:
                        print('Processed {}'.format(row_count))
 
                if data_index not in span_logs:
                    highlight = token_span_util.get_highlight_text(email_id, start_span, end_span)
                    span_logs[data_index] = highlight

                val_dic['highlight'] = span_logs[data_index]

                val_dic['to_do_summary'] = encode_util.decode_sent(summary, back_vocab)

                writer.writerow(val_dic)

            except Exception as inst:
                print('ERROR: {}'.format(inst))




if __name__=='__main__':

    #gen_encode_token_span_dataset(dev = False)

    gen_decode_token_span_dataset()

