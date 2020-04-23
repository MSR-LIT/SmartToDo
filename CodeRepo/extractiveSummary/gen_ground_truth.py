import csv
import pdb
import json
import pickle

if __name__=='__main__':

    path_to_hitApp_logs = '../logs/UHRS_Task_Gold_Pilot1.tsv'
    path_to_ground_truth = '../logs/helpful_UHRS_Task_Gold_Pilot1.pkl'


    hit_logs = {}
    time_spent = []
    with open(path_to_hitApp_logs, encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            judgement = row

            hit_id = judgement['HitID']
            sent_dic = json.loads(judgement['sent_json'])
            data_index = judgement['data_index']
            if hit_id not in hit_logs:
                hit_logs[hit_id] = {}
                hit_logs[hit_id]['to_do_summary'] = []
                hit_logs[hit_id]['task-category'] = []
                hit_logs[hit_id]['subject-useful'] = []
                hit_logs[hit_id]['sent_json'] = {k:[] for k in sent_dic.keys()}
                hit_logs[hit_id]['data_index'] = int(data_index)

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


    # Check emails which have at-least one helpful sentence marked by the judge.
    ground_truth_dic = {}
    count_helpful = 0
    for hit_id in hit_logs.keys():
        data_index = hit_logs[hit_id]['data_index']

        temp = hit_logs[hit_id]['sent_json']


        helpful_set = set()
        for k in temp.keys():
            if temp[k][0] == 'Helpful':
                helpful_set.add(int(k))

        if(len(helpful_set)) > 0:
            ground_truth_dic[data_index] = helpful_set
            count_helpful += 1

    f = open(path_to_ground_truth, "wb")
    pickle.dump(ground_truth_dic, f)
    f.close()

    pdb.set_trace()
    print('Done !')