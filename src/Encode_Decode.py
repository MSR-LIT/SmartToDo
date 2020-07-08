
import pickle
import encode_util
PATH_TO_FWD_DIC = '../data/fwd_vocab.pkl'
PATH_TO_BACK_DIC = '../data/back_vocab.pkl'

def encode(finp, fout, ftype):

    with open(PATH_TO_FWD_DIC, 'rb') as handle:
        fwd_vocab = pickle.load(handle)

    if ftype == 'annotations':
        encode_util.encode_annotation(finp, fout, fwd_vocab)
    elif ftype == 'plain_text':
        encode_util.encode_plain_text(finp, fout, fwd_vocab)
    else:
        raise NotImplementedError


def decode(finp, fout, ftype):

    with open(PATH_TO_BACK_DIC, 'rb') as handle:
        back_vocab = pickle.load(handle)

    if ftype == 'annotations':
            encode_util.decode_annotation(finp, fout, back_vocab)
    elif ftype == 'plain_text':
            encode_util.decode_plain_text(finp, fout, back_vocab)
    else:
            raise NotImplementedError


if __name__=='__main__':

    encode_flag = False
    if encode_flag:

        annotation_file_inp = '../data/Orig_Annotations/SmartToDo_dataset.tsv'

        annotation_file_out = '../data/Coded_Annotations/SmartToDo_dataset.tsv'

        ftype = 'annotations'
        print('Encoding {}...'.format(annotation_file_inp))
        encode(annotation_file_inp, annotation_file_out, ftype)
        print('Done.')

    else:

        annotation_file_inp = '../data/Coded_Annotations/SmartToDo_dataset.tsv'

        annotation_file_out = '../data/Annotations/SmartToDo_dataset.tsv'

        ftype = 'annotations'
        print('Decoding {}...'.format(annotation_file_inp))
        decode(annotation_file_inp, annotation_file_out, ftype)
        print('Done.')






