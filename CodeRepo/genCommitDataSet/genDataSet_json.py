
# Combine avocado_commitment.csv (contains highlighted sentence, highlighted location)
# with Commit_reply_Info.csv, Commit_email_Info.csv and Commit_sentence_Info.csv to obtain a raw dataset.
# This raw data-set needs to be further pruned before annotation using suitable heuristics.

import pdb
import csv
import json
csv.field_size_limit(100000000)

def write_highlights_context(commit_dataset):

    for i in range(len(commit_dataset)):

        data_point = commit_dataset[i]

        data_index = data_point['index']
        fout_highlights = '../data/commit_data/Highlights/{}.txt'.format(data_index)
        with open(fout_highlights, 'w') as f_high:
            f_high.write('C{}:\t{}'.format(data_point['highlight_index'], data_point['highlight']))

        fout_context = '../data/commit_data/Context/{}.txt'.format(data_index)
        with open(fout_context, 'w') as f_con:

            sent_counter = 0
            for past_sent in data_point['reply_to_sentence']:
                f_con.write('P{}:\t{}\n'.format(sent_counter, past_sent))
                sent_counter += 1

            f_con.write('Subject:\t{}\n'.format(data_point['current_subject']))

            sent_counter = 0
            for curr_sent in data_point['current_sentence']:
                f_con.write('C{}:\t{}\n'.format(sent_counter, curr_sent))
                sent_counter += 1


if __name__=='__main__':

        #Load Commit_reply_Info
        path_to_reply_Info_csv = './Commit_reply_Info.csv'
        email_id_reply_info = {}
        email_id_reply_info['NULL'] = {'subject': '', 'sent_to': '', 'sent_from':'', 'body':''}

        with open(path_to_reply_Info_csv, encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:

                email_id = row[0]
                reply_to = row[1]
                if email_id not in email_id_reply_info:
                    email_id_reply_info[email_id] = {}
                    email_id_reply_info[email_id]['reply_to'] = reply_to



        path_to_email_Info_csv = './Commit_email_Info.csv'

        with open(path_to_email_Info_csv, encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:

                email_id = row[0]
                sent_from = row[1]
                sent_to = row[2]
                subject = row[3]
                body = row[4]

                if email_id not in email_id_reply_info:
                    email_id_reply_info[email_id] = {}

                email_id_reply_info[email_id]['sent_from'] = sent_from
                email_id_reply_info[email_id]['sent_to'] = sent_to
                email_id_reply_info[email_id]['subject'] = subject
                email_id_reply_info[email_id]['body'] = body



        #Load Commit_sentence_Info
        path_to_sentence_Info_csv = './Commit_sentence_Info.csv'
        email_id_sentence_info = {}
        email_id_sentence_info['NULL'] = {'sentence':[], 'ner':[]}

        with open(path_to_sentence_Info_csv, encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:

                email_id = row[1]
                sent = row[3]
                ner = row[4]

                if email_id not in email_id_sentence_info:
                    email_id_sentence_info[email_id] = {}
                    email_id_sentence_info[email_id]['sentence'] = [sent]
                    email_id_sentence_info[email_id]['ner'] = [ner]
                else:
                    email_id_sentence_info[email_id]['sentence'].append(sent)
                    email_id_sentence_info[email_id]['ner'].append(ner)



        # Stream through commit data and form json

        path_to_avo_commit_csv= '../../Avo_commitment_data/avocado_commitment.csv'
        count = -1

        dataset = []
        # Extract ids of commitment emails
        data_counter = 0

        with open(path_to_avo_commit_csv, encoding="utf8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:

                count += 1
                if count == 0:
                    continue


                label = int(row[-1])

                if label == 1:
                    data_point = {}

                    email_id = row[1]

                    reply_to_id = email_id_reply_info[email_id]['reply_to']

                    data_point['index'] = data_counter
                    data_point['current_id'] = email_id
                    data_point['reply_to_id'] = reply_to_id

                    data_point['current_sent_from'] = email_id_reply_info[email_id]['sent_from']
                    data_point['reply_to_sent_from'] = email_id_reply_info[reply_to_id]['sent_from']
                    data_point['current_sent_to'] = email_id_reply_info[email_id]['sent_to']
                    data_point['reply_to_sent_to'] = email_id_reply_info[reply_to_id]['sent_to']
                    data_point['current_subject'] = email_id_reply_info[email_id]['subject']
                    data_point['reply_to_subject'] = email_id_reply_info[reply_to_id]['subject']
                    data_point['current_body'] = email_id_reply_info[email_id]['body']
                    data_point['reply_to_body'] = email_id_reply_info[reply_to_id]['body']

                    data_point['highlight'] = row[3]
                    data_point['highlight_index'] = int(row[2][1:])
                    data_point['current_sentence'] = email_id_sentence_info[email_id]['sentence']
                    data_point['current_ner'] = email_id_sentence_info[email_id]['ner']
                    try:
                        data_point['reply_to_sentence'] = email_id_sentence_info[reply_to_id]['sentence']
                    except:
                        data_point['reply_to_sentence'] = []

                    try:
                        data_point['reply_to_ner'] = email_id_sentence_info[reply_to_id]['ner']
                    except:
                        data_point['reply_to_ner'] = []


                    dataset.append(data_point)
                    data_counter += 1

        fout_dataset = '../data/commit_data/Commit_dataset.json'
        with open(fout_dataset, 'w') as f_json:
             json.dump(dataset, f_json)



        # path_to_commit_json = './Commit_dataset.json'
        # with open(path_to_commit_json, encoding="utf8") as f:
        #     for line in f:
        #         data = json.loads(line)

        # data is a list, in only single line dumped json file. [data[0], data[1], ..., data[-1]]

        # write_highlights_context(dataset)
