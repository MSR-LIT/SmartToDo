
import zipfile
import spacy
import csv
import json
import ast
import pdb
from spellchecker import SpellChecker
spell = SpellChecker()

nlp = spacy.load("en_core_web_sm")
PATH_TO_AVOCADO_TEXT = '../data/Avocado/text/'
PATH_TO_OOV_MAPPING = '../data/smarttodo_oov.xlsx'

def get_raw_corpus_text(email_id):

    custodian = email_id[0:3]
    zip_path = PATH_TO_AVOCADO_TEXT + '{}.zip'.format(custodian)

    with zipfile.ZipFile(zip_path) as z:

        filename = '{}/{}.txt'.format(custodian, email_id)
        text = ''
        with z.open(filename) as f:
            for line in f:
                line = line.decode('utf-8').rstrip()

                start = line[0:12]
                if len(line) == 0 or 'Message-ID:' in start or 'Date:' in start or 'In-Reply-To:' in start or 'Reply-To:' in start:
                    continue

                line = line.replace('\"', '\'')
                line = line.replace('\\', '')

                doc = nlp(line, disable=["ner", "parser", "tagger"])
                line = " ".join(str(tok) for tok in doc)

                text += (line + ' ')

    return text




def encode_sent_dev(sent, vocab):

    sent = " ".join(sent.split())
    tokens = nlp(sent, disable=["ner", "parser", "tagger"])

    coded_tokens = []
    for tok in tokens:
        tok = str(tok)
        if tok in vocab:
            coded_tokens.append(vocab[tok])
        elif tok.lower() in vocab:
            coded_tokens.append(vocab[tok.lower()])
        else:
            corrected_tok = spell.correction(tok)
            if corrected_tok in vocab:
                coded_tokens.append(vocab[corrected_tok])
                print('Spell correct : {} -> {}'.format(tok, corrected_tok))
            else:
                coded_tokens.append(tok)
                # coded_tokens.append('UNK')
    coded_str = " ".join(str(coded_tok) for coded_tok in coded_tokens)
    return coded_str




def encode_sent(sent, vocab):

    sent = " ".join(sent.split())
    tokens = nlp(sent, disable=["ner", "parser", "tagger"])
    oov_mapping = _get_oov_mapping()

    coded_tokens = []
    for tok in tokens:
        tok = str(tok)
        if tok in vocab:
            coded_tokens.append(vocab[tok])
        elif tok.lower() in vocab:
            coded_tokens.append(vocab[tok.lower()])
        else:
            corrected_tok = spell.correction(tok)
            if corrected_tok in vocab:
                coded_tokens.append(vocab[corrected_tok])
                print('Spell correct : {} -> {}'.format(tok, corrected_tok))
            elif tok in oov_mapping:
                corrected_tok = oov_mapping[tok]
                corrected_tok_list = corrected_tok.split()
                for corr_tok in corrected_tok_list:
                    if corr_tok in vocab:
                        coded_tokens.append(vocab[corr_tok])
                    elif corr_tok.lower() in vocab:
                        coded_tokens.append(vocab[corr_tok.lower()])
                    elif corr_tok[0] == '#' and corr_tok[-1] == '#':
                        coded_tokens.append(corr_tok)
                    else:
                        coded_tokens.append('UNK')
            else:
                coded_tokens.append('UNK')
    coded_str = " ".join(str(coded_tok) for coded_tok in coded_tokens)
    return coded_str



def decode_sent(sent, vocab):

    tokens = sent.split(" ")
    coded_tokens = []
    for tok in tokens:
        if tok in vocab:
            coded_tokens.append(vocab[tok])
        else:
            coded_tokens.append(tok)

    coded_str = " ".join(coded_tok for coded_tok in coded_tokens)
    return coded_str



def encode_annotation(finp, fout, vocab):


    with open(finp, 'r', encoding='utf-8') as inp_tsv_file, open(fout, 'w', encoding='utf-8', newline='') as out_tsv_file:
        reader = csv.DictReader(inp_tsv_file, delimiter='\t')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(out_tsv_file, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        row_count = 0
        for row in reader:

            row_count += 1
            judgement = row
            highlight = judgement['highlight']
            reply_to_sent_from = judgement['reply_to_sent_from']
            reply_to_sent_to = judgement['reply_to_sent_to']
            reply_to_subject = judgement['reply_to_subject']
            reply_to_body = judgement['reply_to_body']
            current_sent_from = judgement['current_sent_from']
            current_sent_to = judgement['current_sent_to']
            current_subject = judgement['current_subject']
            current_body_before_high = judgement['current_body_before_high']
            current_body_after_high = judgement['current_body_after_high']
            words_json = judgement['words_json']
            sent_json = json.loads(judgement['sent_json'])
            to_do_summary = judgement['to_do_summary']

            #Code content fields

            judgement['highlight'] = encode_sent(highlight, vocab)
            judgement['reply_to_sent_from'] = encode_sent(reply_to_sent_from, vocab)
            judgement['reply_to_sent_to'] =  encode_sent(reply_to_sent_to, vocab)
            judgement['reply_to_subject'] = encode_sent(reply_to_subject, vocab)
            judgement['reply_to_body'] = encode_sent(reply_to_body, vocab)
            judgement['current_sent_from'] = encode_sent(current_sent_from, vocab)
            judgement['current_sent_to'] = encode_sent(current_sent_to, vocab)
            judgement['current_subject'] = encode_sent(current_subject, vocab)
            judgement['current_body_before_high'] = encode_sent(current_body_before_high, vocab)
            judgement['current_body_after_high'] = encode_sent(current_body_after_high, vocab)
            judgement['to_do_summary'] = encode_sent(to_do_summary, vocab)

            coded_sent_json = {}
            for parent_key, parent_val in sent_json.items():
                for child_key, child_val in parent_val.items():
                    coded_str = encode_sent(child_key, vocab)
                    coded_sent_json[parent_key] = {coded_str: child_val}
            judgement['sent_json'] = json.dumps(coded_sent_json)

            candidate_list = ast.literal_eval(words_json)
            coded_candidate_list = []
            for candidate in candidate_list:
                coded_str = encode_sent(candidate, vocab)
                coded_candidate_list.append(coded_str)
            judgement['words_json'] = str(coded_candidate_list)

            # Write coded content to tsv file.
            writer.writerow(judgement)


def decode_annotation(finp, fout, vocab):


    with open(finp, 'r', encoding='utf-8') as inp_tsv_file, open(fout, 'w', encoding='utf-8', newline='') as out_tsv_file:
        reader = csv.DictReader(inp_tsv_file, delimiter='\t')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(out_tsv_file, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        row_count = 0
        for row in reader:

            row_count += 1
            judgement = row
            highlight = judgement['highlight']
            reply_to_sent_from = judgement['reply_to_sent_from']
            reply_to_sent_to = judgement['reply_to_sent_to']
            reply_to_subject = judgement['reply_to_subject']
            reply_to_body = judgement['reply_to_body']
            current_sent_from = judgement['current_sent_from']
            current_sent_to = judgement['current_sent_to']
            current_subject = judgement['current_subject']
            current_body_before_high = judgement['current_body_before_high']
            current_body_after_high = judgement['current_body_after_high']
            words_json = judgement['words_json']
            sent_json = json.loads(judgement['sent_json'])
            to_do_summary = judgement['to_do_summary']

            #Decode content fields

            judgement['highlight'] = decode_sent(highlight, vocab)
            judgement['reply_to_sent_from'] = decode_sent(reply_to_sent_from, vocab)
            judgement['reply_to_sent_to'] =  decode_sent(reply_to_sent_to, vocab)
            judgement['reply_to_subject'] = decode_sent(reply_to_subject, vocab)
            judgement['reply_to_body'] = decode_sent(reply_to_body, vocab)
            judgement['current_sent_from'] = decode_sent(current_sent_from, vocab)
            judgement['current_sent_to'] = decode_sent(current_sent_to, vocab)
            judgement['current_subject'] = decode_sent(current_subject, vocab)
            judgement['current_body_before_high'] = decode_sent(current_body_before_high, vocab)
            judgement['current_body_after_high'] = decode_sent(current_body_after_high, vocab)
            judgement['to_do_summary'] = decode_sent(to_do_summary, vocab)

            coded_sent_json = {}
            for parent_key, parent_val in sent_json.items():
                for child_key, child_val in parent_val.items():
                    coded_str = decode_sent(child_key, vocab)
                    coded_sent_json[parent_key] = {coded_str: child_val}
            judgement['sent_json'] = json.dumps(coded_sent_json)

            candidate_list = ast.literal_eval(words_json)
            coded_candidate_list = []
            for candidate in candidate_list:
                coded_str = decode_sent(candidate, vocab)
                coded_candidate_list.append(coded_str)
            judgement['words_json'] = str(coded_candidate_list)

            # Write decoded content to tsv file.
            writer.writerow(judgement)

def encode_plain_text(finp, fout, vocab):
    raise NotImplementedError

def decode_plain_text(finp, fout, vocab):
    raise NotImplementedError


def _get_oov_mapping():
    import pandas as pd
    oov = pd.read_excel(PATH_TO_OOV_MAPPING)
    clean_oov = oov.loc[:, ['OOV', 'Merged']]
    clean_oov = clean_oov.dropna()
    oov_mapping = dict()
    for index, row in clean_oov.iterrows():
        oov_text = row['OOV']
        merged_text = row['Merged']
        merged_text = merged_text.replace('.', '')
        oov_mapping[oov_text] = merged_text
    return oov_mapping
