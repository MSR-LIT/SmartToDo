import csv
import pdb
import json

if __name__=='__main__':

    # path_to_hitApp_logs = '../logs/UHRS_Task_Team-init-07-16.tsv'
    # path_to_hitApp_logs = '../logs/UHRS_Task_Pilot1-07-30.tsv'
    # path_to_hitApp_logs = '../logs/UHRS_Task_Pilot2-08-02.tsv'
    # path_to_hitApp_logs = '../logs/UHRS_Task_Pilot3-08-09.tsv'
    # path_to_hitApp_logs = '../logs/UHRS_Task_Augment4_08_26.tsv'
    path_to_hitApp_logs = '../logs/UHRS_Task_Augment5_08_30.tsv'

    hit_logs = {}
    time_spent = []
    with open(path_to_hitApp_logs, encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            judgement = row

            hit_id = judgement['HitID']
            data_index = judgement['data_index']
            sent_dic = json.loads(judgement['sent_json'])
            if hit_id not in hit_logs:
                hit_logs[hit_id] = {}
                hit_logs[hit_id]['to_do_summary'] = []
                hit_logs[hit_id]['task-category'] = []
                hit_logs[hit_id]['subject-useful'] = []
                hit_logs[hit_id]['sent_json'] = {k:[] for k in sent_dic.keys()}

            try :
                hit_logs[hit_id]['to_do_summary'].append(judgement['to_do_summary'])
                hit_logs[hit_id]['task-category'].append(judgement['task-category'])
                hit_logs[hit_id]['subject-useful'].append(judgement['subject-useful'])
                for k in hit_logs[hit_id]['sent_json'].keys():
                    candidate, label = next(iter(sent_dic[k].items()))
                    hit_logs[hit_id]['sent_json'][k].append(label)
            except:
                print('Error in loading file !')
                pdb.set_trace()

    # print(hit_logs)

    all_match = 0
    total_candidates = 0
    helpful_match = 0
    helpful_mismatch = 0
    atleast_one_helpful = 0
    match_subject = 0

    # Check emails which have at-least one helpful sentence marked by a judge.

    none_count = 0
    for hit_id in hit_logs.keys():
        print(hit_logs[hit_id]['to_do_summary'])

        none_flag = 0
        for summ in hit_logs[hit_id]['to_do_summary']:
            if summ.lower() == 'none' or summ.lower() == 'none.':
                none_flag = 1
        none_count += none_flag

        # print(hit_logs[hit_id]['sent_json'])
        temp = hit_logs[hit_id]['sent_json']

        total_candidates += len(temp.keys())
        flag_helpful = 0
        try:
            for k in temp.keys():

                    if temp[k][0] == 'Helpful' and temp[k][1] == 'Helpful':
                        helpful_match += 1
                    # elif temp[k][0] == 'Helpful' and temp[k][1] != 'Helpful':
                    elif temp[k][0] == 'Helpful' and temp[k][1] == 'Not Helpful':
                        helpful_mismatch += 1
                    # elif temp[k][1] == 'Helpful' and temp[k][0] != 'Helpful':
                    elif temp[k][1] == 'Helpful' and temp[k][0] == 'Not Helpful':
                        helpful_mismatch += 1


                    if temp[k][0] == temp[k][1]:
                        all_match += 1

                    if temp[k][0] == 'Helpful' or temp[k][1] == 'Helpful':
                        flag_helpful = 1

            if hit_logs[hit_id]['subject-useful'][0] == 'Yes' or hit_logs[hit_id]['subject-useful'][1] == 'Yes':
                flag_helpful = 1

            atleast_one_helpful += flag_helpful


            if hit_logs[hit_id]['subject-useful'][0] == hit_logs[hit_id]['subject-useful'][1]:
                match_subject += 1


        except:

            continue


    print('At least one Helpful (frac) = {:.4f}'.format(atleast_one_helpful/len(hit_logs)))
    print('Absolute agreement (frac) = {:.4f}'.format(all_match/total_candidates))
    print('None summary (frac) = {:.4f}'.format(none_count/len(hit_logs)))
    pdb.set_trace()
    print('Done !')