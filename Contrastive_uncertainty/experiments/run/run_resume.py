# Import general params
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams
from Contrastive_uncertainty.experiments.train.resume_experiments import resume

run_paths = ['nerdk312/practice/2vdhnpef',
            'nerdk312/practice/fdfezneo',
            'nerdk312/practice/30xr80ee',
            'nerdk312/practice/eh6f51ek',
            'nerdk312/practice/3ki677vi']

resume(run_paths, trainer_hparams)
