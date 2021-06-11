# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams

run_paths = ['nerdk312/practice/r6ttdhm3'
            ]

evaluate(run_paths, trainer_hparams)
