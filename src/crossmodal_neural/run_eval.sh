folder=/Users/test/Downloads/nlp_project/math_reference_retrieval/src
cd ${folder}
export PYTHONPATH="${PYTHONPATH}:${folder}"
for domain in stein #trench stein proofwiki stacks
do 
python eval/analyze.py \
    --method tfidf \
    --eval_path ${folder}/crossmodal_neural/output/proofwiki/crossmodal_eval_${domain}.pkl \
    --datapath-base ${folder}/data/data/naturalproofs_${domain}.json \
    --output_path ${folder}/crossmodal_neural/output/proofwiki/crossmodal_analysis_${domain}.pkl 
done 