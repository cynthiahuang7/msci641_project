folder=/Users/test/Downloads/nlp_project/math_reference_retrieval/src
cd ${folder}
export PYTHONPATH="${PYTHONPATH}:${folder}"
for domain in trench stein proofwiki stacks
do 
python classic/baselines.py \
    --method bm25 \
    --datapath ${folder}/data/data/pairwise_${domain}__bert-base-cased.pkl \
    --datapath-base ${folder}/data/data/naturalproofs_${domain}.json \
    --savedir ${folder}/classic/output/bm25_eval_${domain}.pkl \
    --tokenizer bert-base-cased 
done 