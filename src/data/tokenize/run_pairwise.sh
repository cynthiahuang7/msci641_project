folder=/Users/test/Downloads/nlp_project/math_reference_retrieval/src
cd ${folder}
export PYTHONPATH="${PYTHONPATH}:${folder}"
for domain in trench stein 
do 
python data/tokenize/tokenize_pairwise.py --model pairwise \
--model-type tbs17/MathBERT \
--filepath ${folder}/data/data/naturalproofs_${domain}.json \
--output-path ${folder}/data/data \
--dataset-name ${domain} 
done 