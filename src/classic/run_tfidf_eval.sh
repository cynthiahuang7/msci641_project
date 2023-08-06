folder=/Users/test/Downloads/nlp_project/math_reference_retrieval/src
cd ${folder}
export PYTHONPATH="${PYTHONPATH}:${folder}"
for domain in trench stein proofwiki stacks
do 
python eval/analyze.py \
    --method tfidf \
    --eval_path ${folder}/classic/output/tfidf_eval_${domain}.pkl \
    --datapath-base ${folder}/data/data/naturalproofs_${domain}.json \
    --output_path ${folder}/classic/output/tfidf_analysis_${domain}.pkl 
done 
