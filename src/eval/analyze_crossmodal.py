project_dir = '/Users/test/Downloads/nlp_project/math_reference_retrieval/src'
import torch 
import numpy as np
import sys
sys.path.append(project_dir)
from crossmodal_neural.dataset.dataset_string import string_dataset

def get_score(model, tokenizer, encoder, example_str, reference_str): 
    statement1, st1_mask = tokenizer.tokenize_exp_as_symbols(example_str)
    encoded_statement1 = list()
    for token in statement1: 
        if token not in encoder:
            encoded_statement1.append(0)
        else:
            encoded_statement1.append(encoder[token])
    st1_len = len(statement1)        
    statement2, st2_mask = tokenizer.tokenize_exp_as_symbols(reference_str)
    encoded_statement2 = list()
    for token in statement2:
        if token not in encoder:
            encoded_statement2.append(0)
        else:
            encoded_statement2.append(encoder[token])
    st2_len = len(statement2)
    # to tensor 
    encoded_statement1 = torch.tensor(encoded_statement1).unsqueeze(0)
    encoded_statement2 = torch.tensor(encoded_statement2).unsqueeze(0)
    st1_mask = torch.tensor(st1_mask).unsqueeze(0)
    st2_mask = torch.tensor(st2_mask).unsqueeze(0)
    st1_len = torch.tensor(st1_len).unsqueeze(0)
    st2_len = torch.tensor(st2_len).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        model_output = model(encoded_statement1, st1_mask, st1_len, encoded_statement2, st2_mask, st2_len)
    # convert log soft max to probabilities
    model_output = torch.exp(model_output)
    score = model_output.numpy()[0][0]
    return score 

from crossmodal_neural.util import tokenizer
from crossmodal_neural.models.CrossModalEmbedding import SiameseNet
from crossmodal_neural.tasks.preprocess_star_task import PreprocessStarTask
import json

use_proofwiki = True
MAX_LEN = 256 if use_proofwiki else 200
VOCA_SIZE = 17325 if use_proofwiki else 10168
model = SiameseNet(
    512,
    32,
    VOCA_SIZE,
    max_len=MAX_LEN,
    hidden_size=512,
    out_embedding=512,
    device='cpu',
    attention_heads=4,
    word_embedding=512,
)
if use_proofwiki: 
    model.load_state_dict( torch.load(project_dir + '/crossmodal_neural/models/trained_model.pt'))
    print('model loaded')
else: 
    model.load_state_dict( torch.load(project_dir + '/crossmodal_neural/models/trained_model_random.pt'))
model.eval()

if use_proofwiki: 
    filename = project_dir + "/crossmodal_neural/dataset/proofwiki/neg_1_"
else: 
    filename = project_dir + "/crossmodal_neural/dataset/random/neg_1_"
with open(f"{filename}statements.json", "r") as f:
    statements = json.load(f)
data_prep = PreprocessStarTask()
#with Flow("Generating statement representation model - Negatives ") as flow1:
data_prep.prepare_input(statements, max_len=MAX_LEN)

for domain in ['stein']:
    print(domain)
    data_file = project_dir + "/data/data/naturalproofs_" + domain + ".json"
    string_data = string_dataset(data_file,)
    model_output = {}
    # dict_keys(['x2ranked', 'x2rs', 'rids', 'name'])
    model_output['name'] = 'crossmodal'
    # empty integer array
    rids = np.array([], dtype=int)
    for ref in string_data['refs']: 
        rids = np.append(rids, int(ref['metadata']['rid']))
    model_output['rids'] = rids# array of all reference id 
    model_output['x2rs'] = {}
    model_output['x2ranked'] = {}
    for unmatched_record in string_data['test']['unmatched_data']: 
        example_str = unmatched_record['x']
        statement1, st1_mask = tokenizer.tokenize_exp_as_symbols(example_str)
        st1_len = len(statement1)
        ref_ids = unmatched_record['rids']
        example_id = unmatched_record['metadata']['eid']
        model_output['x2rs'][example_id] = ref_ids
        scorelist = []
        for reference_id in rids:  
            reference_str = string_data['test']['rid2r'][reference_id]['r']
            statement2, st2_mask = tokenizer.tokenize_exp_as_symbols(example_str)
            st2_len = len(statement1)
            score = get_score(model, tokenizer, data_prep.encoder, example_str, reference_str)
            scorelist.append([score, int(reference_id)])
        model_output['x2ranked'][example_id] = scorelist
    import pickle
    if use_proofwiki: 
        with open(project_dir + '/crossmodal_neural/output/proofwiki/crossmodal_eval_' + domain + '.pkl', 'wb') as f:
            pickle.dump(model_output, f)
    else:
        with open(project_dir + '/crossmodal_neural/output/crossmodal_eval_' + domain + '.pkl', 'wb') as f:
            pickle.dump(model_output, f)

