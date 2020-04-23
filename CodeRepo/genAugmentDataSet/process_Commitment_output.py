
import pdb
import csv
import numpy as np
csv.field_size_limit(100000000)


if __name__=='__main__':

    path_to_classifier_out = '../data/augment_data/augment_Aether_output.tsv'
    # Extract probabilities
    prob_list = []
    highlight_set = set()
    # First accumulate original sentences by processing AEther input and Aether output.
    with open(path_to_classifier_out, encoding='utf-8-sig') as tsvfile1:
        reader = csv.reader(tsvfile1, delimiter='\t')

        for row in reader:

            prob = float(row[0])
            sent = row[1]
            label = row[2]

            prob_list.append(prob)

            if prob > 0.90:
                highlight_set.add(label)


    # prob_list = np.array(prob_list)
    # pdb.set_trace()
    # print('Done !')

    path_to_classifier_inp = '../data/augment_data/augment_Aether_Input.tsv'
    path_to_highlight_data = '../data/augment_data/highlight_info.tsv'
    count = 0
    with open(path_to_classifier_inp, encoding="utf8", errors='ignore') as tsvfile2:
        reader = csv.reader(tsvfile2, delimiter='\t')

        with open(path_to_highlight_data, 'w', newline='') as csvfile3:
            writer = csv.writer(csvfile3, delimiter='\t')
            writer.writerow(['id', 'email_id', 'sentence_id', 'sentence', 'label'])

            for row in reader:

                    sent = row[0]
                    label = row[1]

                    if label in highlight_set:
                        temp = label.split(',')
                        id = int(temp[0])
                        email_id = temp[1]
                        sent_id = temp[2]
                        dummy = 1
                        writer.writerow([id, email_id, sent_id, sent, dummy])
                        count += 1

                        if count%1000 == 0:
                            print('Processed {}'.format(count))


