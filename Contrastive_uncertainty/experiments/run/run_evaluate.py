import wandb

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams


# Code to obtain run paths from a project and group
run_paths = []
api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py

runs = api.runs(path="nerdk312/evaluation", filters={"config.group": "OOD detection at different scales experiment"})
for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)


'''
run_paths = ['nerdk312/evaluation/p4ojvcp2',
            'nerdk312/evaluation/1r3ikanf',
            'nerdk312/evaluation/3a5kvvb9',
            'nerdk312/evaluation/2elm4qm4',
            'nerdk312/evaluation/3w3suyyz',
            'nerdk312/evaluation/13f0cla1'
            ]

'''
evaluate(run_paths, trainer_hparams)
