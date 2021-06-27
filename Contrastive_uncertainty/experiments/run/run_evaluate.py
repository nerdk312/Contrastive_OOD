import wandb

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams

'''
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



run_paths = ['nerdk312/evaluation/3n5sk5kt',
            'nerdk312/evaluation/3p7vdk47',
            'nerdk312/evaluation/2rv7nnoy',
            'nerdk312/evaluation/37ei38hd',
            'nerdk312/evaluation/1hc3ceov',
            'nerdk312/evaluation/32mq93hy',
            'nerdk312/evaluation/1ijmgxty',
            'nerdk312/evaluation/2c5k58k8',
            'nerdk312/evaluation/3quinaht',
            'nerdk312/evaluation/3j8ixyq4',
            'nerdk312/evaluation/11aj9uuf',
            'nerdk312/evaluation/1ltgj0bj'
            ]


evaluate(run_paths, trainer_hparams)
