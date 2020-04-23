import csv
import pdb
import json

if __name__=='__main__':

    path_to_hitApp_logs = '../logs/UHRS_Task_Gold_Pilot1.tsv'

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

    total_candidates = 0
    count_data_points = 0
    count_helpful = 0
    atleast_one_helpful = 0

    # Check emails which have at-least one helpful sentence marked by the judge.

    for hit_id in hit_logs.keys():
        summary = hit_logs[hit_id]['to_do_summary'][0]
        print(hit_logs[hit_id]['to_do_summary'])
        temp = hit_logs[hit_id]['sent_json']

        summary = summary.lower()
        if summary == 'none' or summary == 'none.':
            continue

        total_candidates += len(temp.keys())
        count_data_points += 1
        flag_helpful = 0

        for k in temp.keys():

            if temp[k][0] == 'Helpful':
                flag_helpful = 1

        count_helpful += flag_helpful

        if hit_logs[hit_id]['subject-useful'][0] == 'Yes':
            flag_helpful = 1

        atleast_one_helpful += flag_helpful

    print('At least one Helpful (frac) = {:.4f}'.format(count_helpful / count_data_points))
    print('At least one Helpful or Subject Yes (frac) = {:.4f}'.format(atleast_one_helpful/count_data_points))

    pdb.set_trace()
    print('Done !')