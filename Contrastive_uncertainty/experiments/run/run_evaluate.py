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
run_paths = ['nerdk312/evaluation/29ck9osj',
            'nerdk312/evaluation/325cu61i',
            'nerdk312/evaluation/110m7bbi',
            'nerdk312/evaluation/qc68v26a',
            'nerdk312/evaluation/1gyve5kb',
            'nerdk312/evaluation/3loivf31',
            'nerdk312/evaluation/1bzi7svu',
            'nerdk312/evaluation/cuyvogrh',
            'nerdk312/evaluation/14x27zqn',
            'nerdk312/evaluation/30qrthyh',
            'nerdk312/evaluation/1dth3ial'
            ]
'''

'''
run_paths = ['nerdk312/evaluation/1dth3ial'
]
'''

evaluate(run_paths, trainer_hparams)
