import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_auc_score

from Contrastive_uncertainty.Moco.moco_callbacks import quickloading, \
                                                         get_fpr, get_pr_sklearn, get_roc_sklearn

class kNN(pl.Callback):
    def __init__(self, Datamodule,K = 10):
        super().__init__()
        self.Datamodule = Datamodule
        self.K = K

        #self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly

    # Outputs the OOD scores only (does not output the curve)
    def on_validation_epoch_end(self,trainer,pl_module):
        self.kNN_calculation(pl_module)

    # Outputs OOD scores aswell as the ROC curve
    def on_test_epoch_end(self,trainer,pl_module):
        self.kNN_calculation(pl_module)

    
    def kNN_calculation(self,pl_module):

        trainFeatures = pl_module.lemniscate.memory.t() #  Obtain samples from memory bank (memory bank is ordered due to indices)
        trainLabels = self.Datamodule.val_dataloader().dataset.tensors[1] # Obtain labels in unshuffled manner
        trainLabels =  trainLabels.to(pl_module.device)
        C = trainLabels.max() + 1
        total = 0
        top1 = 0
        test_loader = self.Datamodule.train_dataloader()
        with torch.no_grad():
            retrieval_one_hot = torch.zeros(self.K,C,device = pl_module.device)
            for batch_idx, (inputs, targets, indices) in enumerate(test_loader):
                assert len(test_loader) > 0, 'loader is empty'
                if isinstance(inputs, tuple) or isinstance(inputs, list):
                    inputs, *aug_inputs = inputs
                    inputs = inputs.to(pl_module.device)
                    targets = targets.to(pl_module.device)
                batchSize = inputs.size(0)
                features = pl_module.callback_vector(inputs)

                dist = torch.mm(features, trainFeatures)

                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                #import ipdb; ipdb.set_trace()
                candidates = trainLabels.view(1,-1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batchSize * self.K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(pl_module.hparams.softmax_temperature).exp_()

                # Nawid - find the probability of the different classes
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            # Nawid - obtain the predictions
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            #import ipdb; ipdb.set_trace()
            correct = predictions.eq(targets.data.view(-1,1))
            # Nawid - update top 1 and top 5 accuracies
            top1 = top1 + correct.narrow(1,0,1).sum().item()
            total += targets.size(0)
        print(top1*100./total)
    
        return top1/total



