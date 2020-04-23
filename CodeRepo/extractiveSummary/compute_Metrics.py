
import pickle
import numpy as np
import pdb

def get_ranked_output(path_to_file):

    alg_rank_dic = {}
    with open(path_to_file, 'r') as fptr:
        for line in fptr.readlines():
            line = line.rstrip().split('\t')
            data_index = int(line[0])
            ranked_tuple = line[1].split(';')
            ranked_list = []
            for item in ranked_tuple:
                item = item.split(',')
                if float(item[1]) == 0:
                    continue
                ranked_list.append(int(item[0]))
            alg_rank_dic[data_index] = ranked_list

    return alg_rank_dic


def compute_rec_at_K(ground_truth_dic, alg_rank_dic):

    rec_arr = []
    for data_index in ground_truth_dic:
        truth_set = ground_truth_dic[data_index]
        K = len(truth_set)
        pred_val = alg_rank_dic[data_index][0:K]
        count_correct = 0
        for val in pred_val:
            if val in truth_set:
                count_correct +=1
        rec = count_correct/K
        rec_arr.append(rec)

    return np.mean(rec_arr)


def compute_at_least_one_helpful(ground_truth_dic, alg_rank_dic, k):

    count_arr = []
    for data_index in ground_truth_dic:
        truth_set = ground_truth_dic[data_index]
        K = k
        pred_val = alg_rank_dic[data_index][0:K]
        flag = 0
        for val in pred_val:
            if val in truth_set:
                flag = 1
                break
        count_arr.append(flag)

    return np.mean(count_arr)


if __name__=='__main__':

    path_to_ground_truth = '../logs/extractSumRes/helpful_UHRS_Task_Gold_Pilot1.pkl'

    # path_to_ranked_sent = '../logs/extractSumRes/bert_pretrained_rank_max_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/bert_pretrained_rank_mean_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/bert_finetuned_pool_1_rank_mask_cls_sep_mean_max_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/bert_finetuned_pool_1_rank_max_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/bert_finetuned_pool_1_rank_mean_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/bert_finetuned_pool_2_rank_max_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/bert_finetuned_pool_2_rank_mean_UHRS_Task_Gold_Pilot1.txt'

    path_to_ranked_sent = '../logs/extractSumRes/fasttext_d50_max_rank_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/fasttext_d50_mean_rank_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/fasttext_d300_max_rank_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/fasttext_d300_mean_rank_UHRS_Task_Gold_Pilot1.txt'
    # path_to_ranked_sent = '../logs/extractSumRes/tf_rank_UHRS_Task_Gold_Pilot1.txt'

    #Load up Ground Truth "Helpful" Sentences
    ground_truth_dic = pickle.load(open(path_to_ground_truth, "rb"))

    alg_rank_dic = get_ranked_output(path_to_ranked_sent)

    rec_at_K = compute_rec_at_K(ground_truth_dic, alg_rank_dic)

    at_least_one_helpful_atmost2 = compute_at_least_one_helpful(ground_truth_dic, alg_rank_dic, k=2)

    at_least_one_helpful_atmost1 = compute_at_least_one_helpful(ground_truth_dic, alg_rank_dic, k=1)

    at_least_one_helpful_atmost3 = compute_at_least_one_helpful(ground_truth_dic, alg_rank_dic, k=3)

    print('Mean Recall at K = {}'.format(rec_at_K))
    print('At least one "Helpful" at K = 1 : {}'.format(at_least_one_helpful_atmost1))
    print('At least one "Helpful" at K = 2 : {}'.format(at_least_one_helpful_atmost2))
    print('At least one "Helpful" at K = 3 : {}'.format(at_least_one_helpful_atmost3))

    # pdb.set_trace()
    # print('Done !')
