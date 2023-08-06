from tqdm import tqdm
from collections import defaultdict
import json
import re

def replace_links(lines):
    def __replace(line):
        matches = re.findall(r'(\[\[([^]]*)\]\])', line)
        for match in matches:
            full, inner = match
            splt = inner.split('|')
            if len(splt) == 1:
                txt = splt[0]
            elif len(splt) == 2:
                txt = splt[1]
            else:
                txt = ''.join(splt[1:])
            if full in line:
                line = line.replace(full, txt)
        return line
    lines_ = [
        __replace(line) for line in lines
    ]
    return lines_

# ---- data tokenization and loading
def _string_examples(split, dataset, title_only=False, content_only=False):
    stats = defaultdict(int)
    examples = split['examples']
    id2thm = {thm['id'] : thm for thm in dataset['theorems']}

    tokenized = []
    for eid, (tid, pix) in tqdm(enumerate(examples), total=len(examples)):
        ex = id2thm[tid]
        proof = ex['proofs'][pix]

        title = '' if content_only else ex['title']
        content = '' if title_only else ' '.join(replace_links(ex['contents']))
        inputs = "%s%s" % (
            title,
            content
        )

        rids = sorted(set(proof['ref_ids']))
        tokenized.append({
            'x': inputs,
            'rids': rids,
            'metadata': {
                'eid': eid
            }
        })

    return tokenized

def _string_refs(split, dataset, title_only=False, content_only=False):
    stats = defaultdict(int)
    rids = split['ref_ids']
    refs = dataset['theorems'] + dataset['definitions'] + dataset['others']

    tokenized = {}
    for ref in tqdm(refs, total=len(refs)):
        if ref['id'] not in rids:
            continue
        title = '' if content_only else ref['title']
        content = '' if title_only else ' '.join(replace_links(ref['contents']))
        inputs = "%s%s" % (
            title,
            content
        )

        tokenized[ref['id']] = {
            'r': inputs,
            'metadata': {
                'rid': ref['id']
            }
        }

    reflist = [tokenized[r] for r in rids]
    return tokenized, reflist

def string_dataset(
    filepath,
    ref_title_only=False,
    ex_title_only=False,
    ref_content_only=False,
    ex_content_only=False
):
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    base = json.load(open(filepath, 'r'))
    dataset, splits = base['dataset'], base['splits']
    tokenized = {}

    print("---- assuming all refs are present in `valid` `ref_ids`")
    print("== tokenizing refs")
    rid2r, reflist = _string_refs(
        splits['valid'], dataset, 
        title_only=ref_title_only,
        content_only=ref_content_only
    )
    tokenized['refs'] = reflist

    for name, split in splits.items():
        print("== tokenizing %s" % name)
        xs = _string_examples(
            split, dataset,
            title_only=ex_title_only,
            content_only=ex_content_only
        )
        tokenized[name] = {
            'rid2r': rid2r,
            'split_ref_ids': split['ref_ids'],
            'unmatched_data': xs
        }

    return tokenized