
import pickle
import encrypt_util

PATH_TO_FWD_DIC = './data/fwd_vocab.pkl'
PATH_TO_BACK_DIC = './data/back_vocab.pkl'


def encrypt(finp, fout, ftype):

    with open(PATH_TO_FWD_DIC, 'rb') as handle:
        fwd_vocab = pickle.load(handle)

    if ftype == 'uhrs':
        encrypt_util.encode_uhrs(finp, fout, fwd_vocab)
    elif ftype == 'plain_text':
        encrypt_util.encode_plain_text(finp, fout, fwd_vocab)
    else:
        raise NotImplementedError


def decrypt(finp, fout, ftype):

    with open(PATH_TO_BACK_DIC, 'rb') as handle:
        back_vocab = pickle.load(handle)

    if ftype == 'uhrs':
            encrypt_util.decode_uhrs(finp, fout, back_vocab)
    elif ftype == 'plain_text':
            encrypt_util.decode_plain_text(finp, fout, back_vocab)
    else:
            raise NotImplementedError


if __name__=='__main__':

    encode = False
    if encode:
        UHRS_file_inp = ['./data/Orig_UHRS_judgements/merged_UHRS.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Gold_Pilot1.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Pilot1-07-30.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Pilot2-08-02.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Pilot3-08-09.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Augment1_08_12.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Augment2_08_16.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Augment3_08_20.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Augment4_08_26.tsv',
                         './data/Orig_UHRS_judgements/UHRS_Task_Augment5_08_30.tsv']

        UHRS_file_out = ['./data/Encrypted_UHRS_judgements/merged_UHRS.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Gold_Pilot1.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Pilot1-07-30.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Pilot2-08-02.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Pilot3-08-09.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment1_08_12.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment2_08_16.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment3_08_20.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment4_08_26.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment5_08_30.tsv']
        ftype = 'uhrs'

        for i in range(len(UHRS_file_inp)):
            print('Encrypting {}...'.format(UHRS_file_inp[i]))
            finp = UHRS_file_inp[i]
            fout = UHRS_file_out[i]
            encrypt(finp, fout, ftype)
            print('Done.')

    else:

        UHRS_file_inp = ['./data/Encrypted_UHRS_judgements/merged_UHRS.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Gold_Pilot1.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Pilot1-07-30.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Pilot2-08-02.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Pilot3-08-09.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment1_08_12.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment2_08_16.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment3_08_20.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment4_08_26.tsv',
                         './data/Encrypted_UHRS_judgements/UHRS_Task_Augment5_08_30.tsv']


        UHRS_file_out = ['./data/UHRS_judgements/merged_UHRS.tsv',
                         './data/UHRS_judgements/UHRS_Task_Gold_Pilot1.tsv',
                         './data/UHRS_judgements/UHRS_Task_Pilot1-07-30.tsv',
                         './data/UHRS_judgements/UHRS_Task_Pilot2-08-02.tsv',
                         './data/UHRS_judgements/UHRS_Task_Pilot3-08-09.tsv',
                         './data/UHRS_judgements/UHRS_Task_Augment1_08_12.tsv',
                         './data/UHRS_judgements/UHRS_Task_Augment2_08_16.tsv',
                         './data/UHRS_judgements/UHRS_Task_Augment3_08_20.tsv',
                         './data/UHRS_judgements/UHRS_Task_Augment4_08_26.tsv',
                         './data/UHRS_judgements/UHRS_Task_Augment5_08_30.tsv']
        ftype = 'uhrs'

        for i in range(len(UHRS_file_inp)):
            print('Decrypting {}...'.format(UHRS_file_inp[i]))
            finp = UHRS_file_inp[i]
            fout = UHRS_file_out[i]
            decrypt(finp, fout, ftype)
            print('Done.')






