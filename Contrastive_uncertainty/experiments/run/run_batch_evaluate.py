import wandb

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.batch.batch_trainer_params import batch_trainer_hparams


api = wandb.Api()
for group_key, group_trainer_hparams in batch_trainer_hparams.items():
    run_paths = []
    # Gets the runs corresponding to a specific filter (particular group)
    runs = api.runs(path="nerdk312/evaluation", filters={"config.group":f'{group_key}'})
    
    for i in range(len(runs)):
        # Joins together the path of the runs which are separated into different parts in a list
        run_path = '/'.join(runs[i].path) # In the format of  ['nerdk312', 'Toy_evaluation', '1s51anu7'] which needs to be combined with a /
        run_paths.append(run_path)
    
    # Evaluate all the runs for a particular group using the particular hparams
    evaluate(run_paths, group_trainer_hparams)
