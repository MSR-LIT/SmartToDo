import pickle
import spacy
from spacy.vocab import Vocab
from spacy.matcher import PhraseMatcher
import zipfile
import pdb

nlp = spacy.load('en_core_web_sm')

PATH_TO_VOCAB = '../data/fwd_vocab.pkl'

PATH_TO_AVOCADO_TEXT = '../data/Avocado/text/'

with open(PATH_TO_VOCAB, 'rb') as handle:
    fwd_vocab = pickle.load(handle)
VOCAB =  Vocab(strings = fwd_vocab.keys())



def get_highlight_span(email_id, highlight):

    email_text = _get_raw_corpus_text(email_id)

    matcher = PhraseMatcher(VOCAB)

    matcher.add('dummy', None, nlp(highlight, disable=["ner", "parser", "tagger"]),
                nlp(highlight[:-1], disable=["ner", "parser", "tagger"]),
                nlp(highlight[:-2], disable=["ner", "parser", "tagger"]))

    doc = nlp(email_text, disable=["ner", "parser", "tagger"])
    matches = matcher(doc)

    start_span = None
    end_span = None
    if len(matches) == 0:
        # print(email_text)
        # print(highlight)
        print('{}: ERROR : {}'.format(email_id, highlight))
    else:
        start_span = matches[-1][1]
        end_span = matches[-1][2]

        if isinstance(start_span, int) == False or isinstance(end_span, int) == False:
            raise Exception
        #span = doc[start_span:end_span]
        #print('{}: {}'.format(email_id, span.text))

    return start_span, end_span



def get_highlight_text(email_id, start_span, end_span):

    email_text = _get_raw_corpus_text(email_id)

    doc = nlp(email_text, disable=["ner", "parser", "tagger"])

    span = doc[start_span:end_span]

    return span.text


def _get_raw_corpus_text(email_id):

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

                if "From:" in start or "To:" in start or "Cc:" in start or "FROM:" in start or "TO:" in start or "CC:" in start:
                    continue

                line = line.replace('\t', ' ')
                line = line.replace('\"', '\'')
                line = line.replace('\\', '')
                line = ' '.join(line.split())


                text += (line + ' ')

                if '---original' in line:
                    break

    return text
