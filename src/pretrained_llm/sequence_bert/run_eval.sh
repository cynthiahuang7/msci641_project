folder=/Users/test/Downloads/nlp_project/math_reference_retrieval/src
cd ${folder}
export PYTHONPATH="${PYTHONPATH}:${folder}"
for model in joint  
do 
for domain in trench stein 
do 
python eval/run_analysis.py --model ${model} \
--codedir ${folder} \
--train-ds-names proofwiki \
--datadir ${folder}/data/data \
--eval-ds-names ${domain} \
--stein-rencs ${folder}/data/other/pairwise__train_proofwiki__eval_stein__test__encs.pt \
--trench-rencs ${folder}/data/other/pairwise__train_proofwiki__eval_trench__test__encs.pt \
--split test \
--outdir ${folder}/pretrained_llm/sequence_bert 
done 
done 
