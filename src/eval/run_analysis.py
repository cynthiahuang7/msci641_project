import getpass
import subprocess
import argparse
import os

current_user = getpass.getuser()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', default='test')
parser.add_argument('--model', choices=['pairwise', 'joint', 'autoregressive'])
parser.add_argument('--model-type', default='bert-base-cased')
parser.add_argument('--model-name', default='bert-base-cased')
parser.add_argument('--generation', action='store_true')

parser.add_argument('--codedir', default='./')
parser.add_argument('--datadir', default='./data')
parser.add_argument('--ckptdir', default='./data/ckpt')
parser.add_argument('--outdir', default='./output')

parser.add_argument('--stein-rencs', default=None, help='for OOD eval with joint')
parser.add_argument('--trench-rencs', default=None, help='for OOD eval with joint')
parser.add_argument('--modeltok2datatok', default=None, help='for combined proofwiki+stacks model')

parser.add_argument('--train-ds-names', nargs='+', default=['stacks', 'proofwiki', 'both'])
parser.add_argument('--eval-ds-names', nargs='+',  default=['stacks', 'proofwiki', 'both', 'trench', 'stein'])
args = parser.parse_args()

codedir = args.codedir
ckptdir = args.ckptdir
datadir = args.datadir
outdir = args.outdir


env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

i = 1
train_ds_names = args.train_ds_names
eval_ds_names = args.eval_ds_names
for training_ds_name in train_ds_names:
    for eval_ds_name in eval_ds_names:
        print("=== (%d/%d)\n\ttraining_ds_name: %s\n\teval_ds_name: %s" % (
            i, len(train_ds_names)*len(eval_ds_names), training_ds_name, eval_ds_name
        ))

        # --- Setup the paths
        ckptpath = os.path.join(ckptdir, '%s_%s.ckpt' % (args.model, training_ds_name))
        datatype = 'pairwise' if args.model == 'pairwise' else 'sequence'
        if args.model_name == 'bert-base-cased':
            datapath = os.path.join(datadir, '%s_%s__bert-base-cased.pkl' % (datatype, eval_ds_name))
        else: 
            datapath = os.path.join(datadir, '%s_%s_tokenized__tbs17_MathBERT_.pkl' % (datatype, eval_ds_name))
        datapath_base = os.path.join(datadir, 'naturalproofs_%s.json' % eval_ds_name)
        # ----

        # Split
        if eval_ds_name == 'trench' or eval_ds_name == 'stein':
            split = 'test'
        else:
            split = args.split

        # Name the evaluation
        prefix = '%s__' % args.model
        method = '%strain_%s__eval_%s__%s' % (prefix, training_ds_name, eval_ds_name, split)
        output_dir = os.path.join(outdir, 'eval', method)

        print('output_dir: %s' % output_dir)
        
        # --- Joint and Autoregressive
        if args.model != 'pairwise':
            cmd = [
                'python', '-u', os.path.join(codedir, 'pretrained_llm/sequence_bert', 'predict.py'),
                '--checkpoint-path', ckptpath,
                '--datapath', datapath,
                '--output-dir', output_dir,
                '--split', split,
                '--set-mode', '0',
                '--rank-use-first', '1'
            ]
            if args.model == 'joint':
                cmd.extend([
                    '--parallel', '1',
                ])

            if training_ds_name == 'both' and eval_ds_name != 'both':
                cmd.extend([
                    '--modeltok2datatok', args.modeltok2datatok
                ])

            if eval_ds_name == 'stein':
                cmd.extend([
                    '--ood-encs', args.stein_rencs
                ])

            if eval_ds_name == 'trench':
                cmd.extend([
                    '--ood-encs', args.trench_rencs
                ])

            if args.generation:
                from copy import deepcopy
                cmd_ = deepcopy(cmd)
                cmd_.extend([
                    '--no-repeat-ngram-size', '0',
                    '--num-beams', '20',
                ])
                print(' '.join(cmd_))
                process = subprocess.Popen(cmd_, env=env)
                process.wait()
            else:
                cmd.extend([
                    '--max-length', '5' if eval_ds_name == 'stacks' else '20',
                    '--num-beams', '1',
                ])
                print(' '.join(cmd))
                process = subprocess.Popen(cmd, env=env)
                process.wait()

        # --- Pairwise
        else:
            # if output_dir not exist create one 
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cmd = [
                'python', '-u', os.path.join(codedir, 'pretrained_llm/pairwise_bert', 'predict.py'),
                '--model-type', args.model_type,
                '--model-name', args.model_name,
                '--method', method,
                '--checkpoint-path', ckptpath,
                '--datapath', datapath,
                '--datapath-base', datapath_base,
                '--output-dir', output_dir,
                '--split', split
            ]
             

            print(' '.join(cmd))
            process = subprocess.Popen(cmd, env=env)
            process.wait()

        
            # Analyze
            cmd = [
                'python', '-u', os.path.join(codedir, 'eval', 'analyze.py'),
                '--method', method,
                '--eval_path',  os.path.join(output_dir, '%s__eval.pkl' % method),
                '--datapath-base', datapath_base,
                '--output_path', os.path.join(output_dir, '%s_analysis_%s.pkl' % (method, training_ds_name))
            ]

            print(' '.join(cmd))
            process = subprocess.Popen(cmd, env=env)
            process.wait()

        i += 1
