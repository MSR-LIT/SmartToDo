
import json
import csv
import pdb


def convert_str_json_fmt(sent):

    sent = sent.replace('\t', ' ')
    sent = sent.replace('\"', '\'')
    sent = sent.replace('\\', '')
    sent = ' '.join(sent.split())   #Replace multiple spaces with single space to avoid tab creation.
    return sent


def update_candidate_list(current_sent_list, highlight_index):

    updated_sent_list = []
    for i in range(len(current_sent_list)):
        if i == highlight_index:
            continue
        else:
            updated_sent_list.append(current_sent_list[i])

    return updated_sent_list


def gen_current_body_before_and_after(current_sent_list, highlight_index):

    current_body_before_list = []
    current_body_after_list = []

    for i in range(len(current_sent_list)):
        sent = current_sent_list[i]
        sent = convert_str_json_fmt(sent)
        if i == highlight_index:
            continue
        elif i < highlight_index:
            current_body_before_list.append(sent)
        elif i > highlight_index:
            current_body_after_list.append(sent)

    current_body_before = ' '.join(current_body_before_list)
    current_body_after = ' '.join(current_body_after_list)
    return current_body_before, current_body_after



def gen_sent_string_list(sent_list):

    fmt_str = '['
    for i in range(len(sent_list)-1):
        sent = sent_list[i]
        sent = convert_str_json_fmt(sent)
        fmt_str += '\"{}\",'.format(sent)

    try:
        fmt_str += '\"{}\"]'.format(sent_list[-1])
    except:
        fmt_str = '["<No Content>"]'
    return fmt_str


def gen_email_body(sent_list):

    if len(sent_list) == 0:
        return '<No Previous Email Content>'
    for i in range(len(sent_list)):
        sent = sent_list[i]
        sent = convert_str_json_fmt(sent)
        sent_list[i] = sent

    return ' '.join(sent_list)


def prune_candidate_list(candidate_list):

    pruned_candidate_list = []
    for candidate in candidate_list:
        candidate = convert_str_json_fmt(candidate)
        num_tokens = len(candidate.split(' '))
        if num_tokens <= 3:
            continue
        else:
            pruned_candidate_list.append(candidate)

    return pruned_candidate_list

if __name__=='__main__':

    path_to_commit_json = '../data/commit_data/Commit_dataset.json'


    # num_points = 4

    counter = 0

    with open(path_to_commit_json, encoding='utf-8') as json_data:
        data = json.load(json_data)
        end_index = len(data)
        start_index = 1321
        max_points = end_index

        # path_to_hitapp_task = 'Pilot3_batch_start_{}_size_{}.tsv'.format(start_index, max_points)
        path_to_hitapp_task = 'UHRS_judge_debug.tsv'

        with open(path_to_hitapp_task, 'w', encoding='utf-8') as f_out:

            f_out.write('data_index\tcurrent_id\thighlight\treply_to_sent_from\treply_to_sent_to\treply_to_subject'
                        '\treply_to_body\tcurrent_sent_from\tcurrent_sent_to\tcurrent_subject'
                        '\tcurrent_body_before_high\tcurrent_body_after_high\twords_json\n')


            for i in range(start_index, end_index):


                reply_to_subject = convert_str_json_fmt(data[i]['reply_to_subject'])
                reply_to_sent_from = convert_str_json_fmt(data[i]['reply_to_sent_from'])
                reply_to_sent_to = convert_str_json_fmt(data[i]['reply_to_sent_to'])

                current_subject = convert_str_json_fmt(data[i]['current_subject'])
                current_sent_from = convert_str_json_fmt(data[i]['current_sent_from'])
                current_sent_to = convert_str_json_fmt(data[i]['current_sent_to'])


                highlight_index = data[i]['highlight_index']
                highlight_sentence = convert_str_json_fmt(data[i]['highlight'])
                current_sent_list = data[i]['current_sentence']
                reply_to_sent_list = data[i]['reply_to_sentence']


                # current_body = gen_email_body(current_sent_list)
                current_body_before, current_body_after = gen_current_body_before_and_after(current_sent_list, highlight_index)
                reply_to_body = gen_email_body(reply_to_sent_list)

                # Remove highlighted sentence from current list of sentences.
                updated_current_sent_list = update_candidate_list(current_sent_list, highlight_index)
                candidate_list = reply_to_sent_list + updated_current_sent_list
                pruned_candidate_list = prune_candidate_list(candidate_list)

                if len(pruned_candidate_list) < 1 or len(pruned_candidate_list) > 20:
                    continue

                fmt_str = gen_sent_string_list(pruned_candidate_list)

                f_out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.
                            format(data[i]['index'], data[i]['current_id'], highlight_sentence, reply_to_sent_from, reply_to_sent_to,
                                   reply_to_subject, reply_to_body, current_sent_from, current_sent_to, current_subject,
                                   current_body_before, current_body_after, fmt_str))
                counter += 1

                if counter == max_points:
                    break



    print(counter)