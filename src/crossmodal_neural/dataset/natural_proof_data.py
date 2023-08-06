

import random
import json
from crossmodal_neural.dataset.dataset_string import string_dataset

project_dir = '/Users/test/Downloads/nlp_project/math_reference_retrieval/src'
domain = 'proofwiki'
data_file = project_dir + "/data/data/naturalproofs_" + domain + ".json"
string_data = string_dataset(data_file,)
all_reference_index = list(string_data['train']['rid2r'].keys())
statements = dict()
train = dict()
dev = dict()
test = dict()
ids_ex = dict()
ids_ref = dict()
current_index = 0
example_index = 0
dev_index = 0
test_index = 0
for record  in string_data['train']['unmatched_data']:
    example = record['x']
    example_id = record['metadata']['eid']
    if example_id in ids_ex.keys():
        continue
    else: 
        ids_ex[example_id] = current_index
        statements[current_index] = example
        current_index += 1
    # positive example : sample from record['rids']
    positive_ref = random.choice(record['rids'])
    positive_ref_str = string_data['train']['rid2r'][positive_ref]['r']
    positive_ref_id = string_data['train']['rid2r'][positive_ref]['metadata']['rid']
    if positive_ref_id in ids_ref.keys():
        continue
    else: 
        ids_ref[positive_ref_id] = current_index
        statements[current_index] = positive_ref_str
        current_index += 1
    # negative example : sample from all records except record['rids']
    negative_ref = random.choice(list(set(all_reference_index) - set(record['rids'])))
    negative_ref_str = string_data['train']['rid2r'][positive_ref]['r']
    negative_ref_id = string_data['train']['rid2r'][negative_ref]['metadata']['rid']
    if negative_ref_id in ids_ref.keys():
        continue
    else: 
        ids_ref[negative_ref_id] = current_index
        statements[current_index] = negative_ref_str
        current_index += 1
    if example_index <= 7500: 
        train[example_index] =  {"s1": ids_ex[example_id], "s2": ids_ref[positive_ref_id], "score": 1}
        example_index +=1
        train[example_index] =  {"s1": ids_ex[example_id], "s2": ids_ref[negative_ref_id], "score": 0}
        example_index += 1
    elif dev_index <= 600: 
        dev[dev_index] =  {"s1": ids_ex[example_id], "s2": ids_ref[positive_ref_id], "score": 1}
        dev_index +=1
        dev[dev_index] =  {"s1": ids_ex[example_id], "s2": ids_ref[negative_ref_id], "score": 0}
        dev_index += 1
    else: 
        test[test_index] =  {"s1": ids_ex[example_id], "s2": ids_ref[positive_ref_id], "score": 1}
        test_index +=1
        test[test_index] =  {"s1": ids_ex[example_id], "s2": ids_ref[negative_ref_id], "score": 0}
        test_index += 1
print(len(statements))
print(len(train))
print(len(dev))
print(len(test))

# save statements (dictionary) as json file 
with open(project_dir + '/crossmodal_neural/dataset/proofwiki/neg_1_statements.json', 'w') as json_file:
    json.dump(statements, json_file)
with open(project_dir + '/crossmodal_neural/dataset/proofwiki/neg_1_train.json', 'w') as json_file:
    json.dump(train, json_file)
with open(project_dir + '/crossmodal_neural/dataset/proofwiki/neg_1_dev.json', 'w') as json_file:
    json.dump(dev, json_file)
with open(project_dir + '/crossmodal_neural/dataset/proofwiki/neg_1_test.json', 'w') as json_file:
    json.dump(test, json_file)

