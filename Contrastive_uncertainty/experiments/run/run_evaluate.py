import wandb

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams


# Code to obtain run paths from a project and group
run_paths = []
api = wandb.Api()
# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
#https://github.com/wandb/client/blob/v0.12.1/wandb/apis/public.py#L752-L851

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Confusion Log Probability Evaluation"})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"}) # "OOD detection at different scales experiment" (other group I use to run experiments)
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]})
# Choose specifcally the specific group, the CIFAR100 dataset as well as choosing Moco or Supcon model
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.dataset": "CIFAR100","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]})
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","$or": [{"config.dataset":"CIFAR100" }, {"config.dataset":"CIFAR10"}],"$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]})

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco" }, {"config.model_type": "CE"}]})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type": "SupCon"}]})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats"})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco"}, {"config.model_type": "SupCon"}]})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Separate branch combinations","config.branch_weights":[0,0,1]})

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type":"CE"}, {"config.dataset": "MNIST"}, {"config.dataset": "FashionMNIST"}]})


for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)


#"Group: Separate branch combinations"
#"OOD detection at different scales experiment"
#"OOD hierarchy baselines"


#run_paths = ['nerdk312/evaluation/mtb7idxd']
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
'''

evaluate(run_paths, trainer_hparams)
