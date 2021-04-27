from Contrastive_uncertainty.unsup_con_memory.train.train_unsup_con_memory import train
from Contrastive_uncertainty.unsup_con_memory.config.unsup_con_memory_params import unsup_con_memory_hparams


unsup_con_memory_hparams['bsz'] = 32
unsup_con_memory_hparams['instance_encoder'] = 'resnet18'
unsup_con_memory_hparams['epochs'] = 1
unsup_con_memory_hparams['fast_run'] = True
unsup_con_memory_hparams['num_negatives'] = 16
unsup_con_memory_hparams['training_ratio'] = 0.01
unsup_con_memory_hparams['validation_ratio'] = 0.2
unsup_con_memory_hparams['test_ratio'] = 0.2
unsup_con_memory_hparams['val_check'] = 1
unsup_con_memory_hparams['project'] = 'practice'  # evaluation, contrastive_training
unsup_con_memory_hparams['pretrained_network'] = None  # 'Pretrained_models/finetuned_network.pt'
unsup_con_memory_hparams['quick_callback'] = True
train(unsup_con_memory_hparams)
