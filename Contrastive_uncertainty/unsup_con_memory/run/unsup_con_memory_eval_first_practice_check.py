from Contrastive_uncertainty.general_clustering.train.train_general_clustering import train
from Contrastive_uncertainty.unsup_con_memory.config.unsup_con_memory_params import unsup_con_memory_hparams
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_module import UnSupConMemoryModule
from Contrastive_uncertainty.unsup_con_memory.models.unsup_con_memory_model_instance import ModelInstance


unsup_con_memory_eval_hparams['bsz'] = 32
unsup_con_memory_eval_hparams['instance_encoder'] = 'resnet18'
unsup_con_memory_eval_hparams['epochs'] = 1
unsup_con_memory_eval_hparams['fast_run'] = True
unsup_con_memory_eval_hparams['num_negatives'] = 16
unsup_con_memory_eval_hparams['training_ratio'] = 0.01
unsup_con_memory_eval_hparams['validation_ratio'] = 0.2
unsup_con_memory_eval_hparams['test_ratio'] = 0.2
unsup_con_memory_eval_hparams['val_check'] = 1
unsup_con_memory_eval_hparams['project'] = 'practice'  # evaluation, contrastive_training
#unsup_con_memory_eval_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
unsup_con_memory_eval_hparams['quick_callback'] = True
evaluate(unsup_con_memory_eval_hparams)
