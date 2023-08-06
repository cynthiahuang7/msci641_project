folder=/Users/test/Downloads/nlp_project/math_reference_retrieval/src
cd ${folder}
export PYTHONPATH="${PYTHONPATH}:${folder}"
for model in pairwise  
do 
for domain in trench stein 
do 
python eval/run_analysis.py --model ${model} \
--codedir ${folder} \
--train-ds-names proofwiki \
--datadir ${folder}/data/data \
--eval-ds-names ${domain} \
--split test \
--outdir ${folder}/pretrained_llm/pairwise_bert 
done 
done 

