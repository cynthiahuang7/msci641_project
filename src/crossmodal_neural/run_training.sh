folder=/Users/test/Downloads/nlp_project/math_reference_retrieval/src
cd ${folder}
export PYTHONPATH="${PYTHONPATH}:${folder}"
python crossmodal_neural/train_star_flow.py --num_negatives=1 --use_proofwiki
