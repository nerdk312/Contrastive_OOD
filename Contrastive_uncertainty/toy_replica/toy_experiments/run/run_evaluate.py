import wandb

# Import general params
from Contrastive_uncertainty.toy_replica.toy_experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.toy_replica.toy_experiments.config.trainer_params import trainer_hparams

run_paths = []
api = wandb.Api()
# Gets the runs corresponding to a specific filter
'''
runs = api.runs(path="nerdk312/Toy_evaluation", filters={"config.group": "Toy Group practice"})
for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path) # In the format of  ['nerdk312', 'Toy_evaluation', '1s51anu7'] which needs to be combined with a /
    run_paths.append(run_path)
'''

# import ipdb; ipdb.set_trace()
#run_paths = ['nerdk312/Toy_evaluation/1mutstsx']
run_paths = ['nerdk312/Toy_evaluation/1f917a5g']

evaluate(run_paths, trainer_hparams)
