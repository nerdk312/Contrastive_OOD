# Import general params
from Contrastive_uncertainty.toy_replica.toy_experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.toy_replica.toy_experiments.config.trainer_params import trainer_hparams

run_paths = ['nerdk312/practice/26atoumj',
    'nerdk312/practice/ob88dgqo',
    'nerdk312/practice/2beu9lsn']

evaluate(run_paths, trainer_hparams)
