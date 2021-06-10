# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams

run_paths = ['nerdk312/practice/2287a5ve',
            'nerdk312/practice/3kia1igc',
            'nerdk312/practice/32lz34l2'
            ]

evaluate(run_paths,trainer_hparams)
