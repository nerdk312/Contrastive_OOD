import wandb

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams


# Code to obtain run paths from a project and group
run_paths = []
api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py


# Second part used of the filter used for the purpose of an or statement to calculate the different values (Shown in the github link above )
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]})

#"config.model_type":"Moco",'config.model_type':"SupCon"
#}) # "OOD detection at different scales experiment" (other group I use to run experiments)
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Separate branch combinations","config.branch_weights":[0,0,1]})
for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)

#import ipdb; ipdb.set_trace()

