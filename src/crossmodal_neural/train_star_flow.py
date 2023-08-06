#from prefect.engine.results import LocalResult
from crossmodal_neural.tasks.preprocess_star_task import PreprocessStarTask
from crossmodal_neural.tasks.training_star_task import TrainingTaskStar
from loguru import logger
import argparse
import json
#from dynaconf import settings
import sys
import os

parser = argparse.ArgumentParser(description="Training model")
parser.add_argument("--use_proofwiki", help="use proofwiki examples", action="store_true")
parser.add_argument("--num_negatives", type=int)
parser.add_argument("--use_random", help="use random examples", action="store_true")
parser.add_argument("--output_log", type=str, help=f"output log file", default="log_execution.json")
parser.add_argument("--output_model", type=str, help=f"output model file", default="trained_model.pt")
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--out_embedding", type=int, default=512)
parser.add_argument("--embedding", type=int, default=512)
parser.add_argument("--decay", type=float, default=0.01)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--att_head", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=2e-4)

args = parser.parse_args()

use_proofwiki = args.use_proofwiki 
use_random = args.use_random

if use_proofwiki and use_random:
    logger.error(f"You can only use one of these options: use_proofwiki or use_random, you are using both!")
    sys.exit()

if not use_proofwiki and not use_random:
    logger.error(f"You didn't set use_proofwiki or use_random, using random examples as default.")
    

num_negatives = args.num_negatives

if num_negatives not in [1,2,5,10]:
    logger.error(f"Number of negatives should be 1, 2, 5 or 10, you are selecting: {num_negatives}")
    sys.exit()

output_log = args.output_log
output_model = args.output_model
hidden_size = args.hidden_size

out_embedding = args.out_embedding
embedding = args.embedding
decay = args.decay
max_len = args.max_len
batch_size = args.batch_size
att_head = args.att_head
learning_rate = args.learning_rate

if use_proofwiki:
    filename = f"./crossmodal_neural/dataset/proofwiki/neg_{num_negatives}_"
else:
    filename = f"./crossmodal_neural/dataset/random/neg_{num_negatives}_"

testing = False 
logger.info(f"Loading files starting with: ./dataset/random/neg_{num_negatives}_")
with open(f"{filename}statements.json", "r") as f:
    statements = json.load(f)
    logger.info(f"Statements file: {filename}statements.json")
with open(f"{filename}train.json", "r") as f:
    train = json.load(f)
    if testing:
        # train is a dictionary,  only keep 100 keys
        train = {k: train[k] for k in list(train)[:100]} 
    logger.info(f"Train file: {filename}train.json")
with open(f"{filename}test.json", "r") as f:
    test = json.load(f)
    if testing:
        test = {k: test[k] for k in list(test)[:100]} 
    logger.info(f"Test file: {filename}test.json")
with open(f"{filename}dev.json", "r") as f:
    dev = json.load(f)
    if testing:
        dev = {k: dev[k] for k in list(dev)[:100]} 
    logger.info(f"Dev file: {filename}dev.json")



#CACHE_LOCATION = settings["cache_location"]
#cache_args = dict(
#    target="{task_name}--"+f"{num_negatives}--{use_random}.pkl",
#    checkpoint=True
#)

if not os.path.exists('./crossmodal_neural/models'):
        os.mkdir('./crossmodal_neural/models')

if not os.path.exists('./crossmodal_neural/logs'):
        os.mkdir('./crossmodal_neural/logs')

data_prep = PreprocessStarTask()
## Param
MAX_LEN = max_len
#with Flow("Generating statement representation model - Negatives ") as flow1:
dataset = data_prep.run(train, test, dev, statements, max_len=MAX_LEN)
print('========== train_task ==========')

train_task = TrainingTaskStar()
# Param
BATCH_SIZE = batch_size
NUM_EPOCHS = 32
LEARNING_RATE = learning_rate
ATTENTION_HEAD = att_head

print(dataset["vocab"])
MAX_LEN = 200 if use_random else 256

train_task.run(
    dataset["train"],
    dataset["test"],
    dataset["dev"],
    num_negatives=num_negatives,
    output_log=output_log,
    output_model=output_model,
    vocab_size=dataset["vocab"],
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    max_sequence_len=MAX_LEN,
    learning_rate=LEARNING_RATE,
    hidden_size=hidden_size,
    out_embedding=out_embedding,
    attention_heads=ATTENTION_HEAD,
    word_embedding=embedding,
    decay=decay,
    )

