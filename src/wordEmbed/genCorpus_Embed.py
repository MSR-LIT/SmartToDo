
import spacy
import csv
import pdb

import time

def preProcess_str(sent):

    sent = sent.replace('\t', ' ')
    sent = ' '.join(sent.split())   #Replace multiple spaces with single space to avoid tab creation.
    return sent


if __name__=='__main__':

        nlp = spacy.load("en_core_web_sm")

        #Load Sentences from CSV file
        path_to_raw_sentences = '../../data/wordEmbed_data/raw_Avocado_Sentences.csv'

        path_to_gensim_input = '../../data/wordEmbed_data/Tokenized_Sentences.txt'
        with open(path_to_gensim_input,'w') as f_sent:
            start = time.time()

            # Tokenize with spacy. Replace PERSON with  #name.
            with open(path_to_raw_sentences, encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                count = 0
                for row in reader:
                    sent = row[0]
                    sent = preProcess_str(sent)

                    doc = nlp(sent)
                    tok_list = []
                    for tok in doc:
                        if tok.ent_type_ == 'PERSON':
                            tok_list.append('#name')
                        else:
                            tok_list.append(str(tok))
                    count += 1
                    f_sent.write(' '.join(tok_list)+'\n')

                    if count % 1000 == 0:
                        finish = time.time()
                        print('Processed = {} , Elapsed = {:.4f} s'.format(count, finish-start))
                        start = time.time()


        print('Generating Corpus ... Done !')
