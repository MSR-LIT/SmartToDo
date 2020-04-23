
#Obtain ids and reply-to relationships for commitment emails

import pdb
import csv
csv.field_size_limit(100000000)

path_to_highlight_data = '../data/augment_data/highlight_info.tsv'
count = -1
pos_label = 0

# Extract ids of commitment emails
commit_id_list = []
with open(path_to_highlight_data, encoding="utf8") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:

        count += 1
        if count == 0:
            continue

        label = int(row[-1])

        if label == 1:
            commit_id_list.append(row[1])
            pos_label += 1


# Generate SQL query to obtain reply info about commitment emails.

contain_string = '('
for i in range(len(commit_id_list)-1):
    contain_string += '\''+commit_id_list[i]+'\''+', '

contain_string += '\''+commit_id_list[-1]+'\''+')'


sql_query_f_name = 'getCommit_reply_Info.sql'

with open(sql_query_f_name, 'w') as f_sql:
    f_sql.write('SELECT id, relationships_reply_to FROM Emails \n')
    f_sql.write('WHERE id in '+contain_string)
