import wandb

# Import general params
from Contrastive_uncertainty.experiments.train.update_experiments import update
from Contrastive_uncertainty.experiments.config.update_params import update_hparams

'''
# Code to obtain run paths from a project and group
run_paths = []
api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Separate branch combinations","config.branch_weights":[0,0,1]}) # "OOD detection at different scales experiment" (other group I use to run experiments)
for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)

#import ipdb; ipdb.set_trace()
#"Different branch weights"
#"Group: Separate branch combinations"
#"OOD detection at different scales experiment"
#"OOD hierarchy baselines"
'''

run_paths = ['nerdk312/evaluation/1r5iu5j5',
            'nerdk312/evaluation/23z052sj',
            'nerdk312/evaluation/24i7kk0f',
            'nerdk312/evaluation/bdw3yqb8',
            'nerdk312/evaluation/3m3vyk1p',
            'nerdk312/evaluation/339rxusg'
            ]


update(run_paths, update_hparams)
