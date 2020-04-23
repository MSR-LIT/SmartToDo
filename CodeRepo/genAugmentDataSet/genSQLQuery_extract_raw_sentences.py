
#Obtain ids from which highlighted sentence is already available. We will exclude these emails and search in other emails
# for commitment sentences.

import pdb
import csv
csv.field_size_limit(100000000)
path_to_avo_commit_csv= '../../Avo_commitment_data/avocado_commitment.csv'
count = -1
pos_label = 0

# Extract ids of commitment emails
commit_id_list = []
with open(path_to_avo_commit_csv, encoding="utf8") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:

        count += 1
        #Exclude the header line.
        if count == 0:
            continue

        label = int(row[-1])

        pos_label += label

        if label == 1:
            commit_id_list.append(row[1])



# Generate SQL query to obtain raw sentences other than the those emails already obtained.

contain_string = '('
for i in range(len(commit_id_list)-1):
    contain_string += '\''+commit_id_list[i]+'\''+', '

contain_string += '\''+commit_id_list[-1]+'\''+')'


sql_query_f_name = 'get_augment_raw_Sentences.sql'

with open(sql_query_f_name, 'w') as f_sql:
    f_sql.write('SELECT s.id, email_id, sentence_index, sentence FROM Sentences s, Emails e \n')
    f_sql.write('WHERE e.id = s.email_id AND e.relationships_duplicate_of IS NULL '
                'AND S.token_count > 2 AND S.token_count <= 50 AND S.email_i '+contain_string)