# Import general params
from Contrastive_uncertainty.toy_replica.toy_experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.toy_replica.toy_experiments.config.trainer_params import trainer_hparams

run_paths = ['nerdk312/Toy_evaluation/2x5bww4f']

evaluate(run_paths, trainer_hparams)
