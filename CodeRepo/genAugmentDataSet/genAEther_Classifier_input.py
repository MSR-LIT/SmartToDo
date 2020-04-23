
import csv
import pdb


def preProcess(sent):

    sent = sent.replace('\t', ' ')
    sent = sent.replace('\n', '')
    sent = sent.replace('\"', '\'')
    sent = ' '.join(sent.split())   #Replace multiple spaces with single space to avoid tab creation.
    return sent


if __name__=='__main__':

    #Load raw Avocado Sentences from CSV file
    path_to_raw_sentences = '../data/augment_data/raw_Highlight_Candidates.csv'

    path_to_Aether_input = '../data/augment_data/augment_Aether_Input.tsv'
    # path_to_Aether_input = '../data/augment_data/augment_Aether_Input_10k.tsv'


    with open(path_to_raw_sentences, encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        with open(path_to_Aether_input, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')

            count = 0
            for row in reader:
                sent_id = row[0]
                email_id = row[1]
                sent_index = row[2]
                sent = row[-1]
                sent = preProcess(sent)
                writer.writerow([sent, '{},{},{}'.format(sent_id,email_id,sent_index)])
                count += 1

                if count%10000 == 0:
                    print('Processed {}'.format(count))