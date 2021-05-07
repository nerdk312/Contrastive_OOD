import wandb
# Import evaluation methods 
from Contrastive_uncertainty.cross_entropy.evaluate.evaluate_cross_entropy import evaluation as CE_evaluation
from Contrastive_uncertainty.Contrastive.evaluate.evaluate_contrastive import evaluation as Moco_evaluation
from Contrastive_uncertainty.sup_con.evaluate.evaluate_sup_con import evaluation as Supcon_evaluation

from Contrastive_uncertainty.PCL.evaluate.evaluate_pcl import evaluation as PCL_evaluation
from Contrastive_uncertainty.unsup_con_memory.evaluate.evaluate_unsup_con_memory import evaluation as UnSupConMemory_evaluation

acceptable_single_models = ['CE','Moco','SupCon','PCL','UnSupConMemory']

# Dict for the model name, parameters and specific training loop
model_dict = {'CE':{'evaluate':CE_evaluation},                
                'Moco':{'evaluate':Moco_evaluation},
                'SupCon':{'evaluate':Supcon_evaluation},
                'PCL':{'evaluate':PCL_evaluation},                
                'UnSupConMemory':{'evaluate':UnSupConMemory_evaluation}
                }


# list of run paths for evaluate
run_paths = ['nerdk312/practice/1zvouy7b']

for run_path in run_paths:
    api = wandb.Api()    
    previous_run = api.run(path=run_path)
    previous_config = previous_run.config
    model_type = previous_config['model_type']
    
    # Selects the previous model and then performs the evaluate function
    assert model_type in acceptable_single_models, 'model type response not in list of acceptable responses'
    model_dict[model_type]['evaluate'](run_path)
    '''
    try:
        model_dict[model_type]['evaluate'](run_path)
    except:
        print('f{run_path} evaluation did not work')
    '''