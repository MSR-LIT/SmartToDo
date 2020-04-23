import pdb
import csv
csv.field_size_limit(100000000)

# Obtain sentence query for email ids

path_to_reply_Info_csv = './Commit_reply_Info.csv'

# Extract email ids of interest, either the commitment email or its reply-to email
email_id_set = set()
with open(path_to_reply_Info_csv, encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:

        email_id_set.add(row[0])
        email_id_set.add(row[1])

email_id_set.remove('NULL')
print('Id extraction Done !')

email_id_list = list(email_id_set)
# Generate SQL query to obtain sentences in the email ids of interest.

contain_string = '('
for i in range(len(email_id_list)-1):
    contain_string += '\''+email_id_list[i]+'\''+', '

contain_string += '\''+email_id_list[-1]+'\''+')'

sql_query_f_name = 'getCommit_sentence_Info.sql'

with open(sql_query_f_name, 'w') as f_sql:
    f_sql.write('SELECT id, email_id, sentence_index, sentence, ner_tags, token_count FROM Sentences \n')
    f_sql.write('WHERE email_id in '+contain_string+'\n')
    f_sql.write('ORDER BY id')


sql_query_f_name = 'getCommit_email_Info.sql'

with open(sql_query_f_name, 'w') as f_sql:
    f_sql.write('SELECT id, outlook_sender_name, sentto_address, processed_subject, body FROM Emails \n')
    f_sql.write('WHERE id in '+contain_string)
