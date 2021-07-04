import wandb

# Import general params
from Contrastive_uncertainty.experiments.train.update_experiments import update # Method which can be used in any case
from Contrastive_uncertainty.toy_replica.toy_experiments.config.update_params import update_hparams

'''
# Code to obtain run paths from a project and group
run_paths = []
api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Group: Separate branch combinations"}) # "OOD detection at different scales experiment" (other group I use to run experiments)
for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)
'''

#"Group: Separate branch combinations"
#"OOD detection at different scales experiment"
#"OOD hierarchy baselines"

run_paths = ['nerdk312/Toy_evaluation/2xyccgvz']


update(run_paths, update_hparams)
