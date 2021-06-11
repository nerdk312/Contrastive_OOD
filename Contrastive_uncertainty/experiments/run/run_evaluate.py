# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams

run_paths = ['nerdk312/evaluation/30qrthyh',
            'nerdk312/evaluation/1bzi7svu',
            'nerdk312/evaluation/qc68v26a'
            ]

evaluate(run_paths, trainer_hparams)
