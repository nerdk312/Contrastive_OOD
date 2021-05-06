import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import re

# Obtained from https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py
def binary_cross_entropy(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.)

    return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)

class BCELoss(nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=False):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(input, target,
                                    weight=self.weight, reduction=self.reduction,
                                    smooth_eps=self.smooth_eps, from_logits=self.from_logits)

def label_smoothing(ys,smoothing,num_classes): # Based on equation 1 from https://amaarora.github.io/2020/07/18/label-smoothing.html#fastaipytorch-implementation-of-label-smoothing-cross-entropy-loss
    import ipdb; ipdb.set_trace()
    soft_target = torch.where(ys>0,ys,smoothing/num_classes) #  Anything that fails to meet the condition is changed, gives values to the off diagonal terms
    soft_target = torch.where(soft_target<=smoothing/num_classes, soft_target , 1-smoothing +(smoothing/num_classes)) # calculates the value for the target label
    return soft_target

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

# https://amaarora.github.io/2020/07/18/label-smoothing.html#fastaipytorch-implementation-of-label-smoothing-cross-entropy-loss - Implementation of label smoothing
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss 

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction
    
    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # (1-ε)* H(q,p) + ε*H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 

# Altered version of Wandb confusion matrix to allow checking the accuracy of the OOD examples
def OOD_conf_matrix(probs=None, y_true=None, preds=None, class_names=None,OOD_class_names = None):
    """
    Computes a multi-run confusion matrix.

    Arguments:
        probs (2-d arr): Shape [n_examples, n_classes]
        y_true (arr): Array of label indices.
        preds (arr): Array of predicted label indices.
        class_names (arr): Array of class names.
        OOD_class_names (arr): Array of class names.

    Returns:
        Nothing. To see plots, go to your W&B run page then expand the 'media' tab
        under 'auto visualizations'.

    Example:
        ```
        vals = np.random.uniform(size=(10, 5))
        probs = np.exp(vals)/np.sum(np.exp(vals), keepdims=True, axis=1)
        y_true = np.random.randint(0, 5, size=(10))
        labels = ["Cat", "Dog", "Bird", "Fish", "Horse"]
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs, y_true=y_true, class_names=labels)})
        ```
    """
    '''
    np = util.get_module(
        "numpy",
        required="confusion matrix requires the numpy library, install with `pip install numpy`",
    )
    '''
    # change warning
    assert probs is None or len(probs.shape) == 2, (
        "confusion_matrix has been updated to accept"
        " probabilities as the default first argument. Use preds=..."
    )

    assert (probs is None or preds is None) and not (
        probs is None and preds is None
    ), "Must provide probabilties or predictions but not both to confusion matrix"

    if probs is not None:
        preds = np.argmax(probs, axis=1).tolist()

    assert len(preds) == len(
        y_true
    ), "Number of predictions and label indices must match"

    if class_names is not None:
        n_classes = len(class_names)
        class_inds = set(preds).union(set(y_true))
        assert max(preds) <= len(
            class_names
        ), "Higher predicted index than number of classes"
        assert max(y_true) <= len(
            class_names
        ), "Higher label class index than number of classes"
    else:
        class_inds = set(preds).union(set(y_true))
        n_classes = len(class_inds)
        class_names = ["Class_{}".format(i) for i in range(1, n_classes + 1)]

    if OOD_class_names is not None:
        OOD_n_classes = len(OOD_class_names)
        OOD_class_inds = set(preds).union(set(y_true))
        assert max(preds) <= len(
            OOD_class_names
        ), "Higher predicted index than number of classes"
        assert max(y_true) <= len(
            OOD_class_names
        ), "Higher label class index than number of classes"
    else:
        OOD_class_inds = set(preds).union(set(y_true))
        OOD_n_classes = len(class_inds)
        OOD_class_names = ["Class_{}".format(i) for i in range(1, n_classes + 1)]


    # get mapping of inds to class index in case user has weird prediction indices
    class_mapping = {}
    for i, val in enumerate(sorted(list(class_inds))):
        class_mapping[val] = i

    counts = np.zeros((n_classes, n_classes))
    for i in range(len(preds)):
        counts[class_mapping[y_true[i]], class_mapping[preds[i]]] += 1

    data = []
    for i in range(n_classes):
        for j in range(n_classes):
            data.append([OOD_class_names[i], class_names[j], counts[i, j]]) # Data takes in the name of the ith class (for actual), name of jth class for OOD prediction and the actual data which is counts n,j to make the confusion matrix

    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }

    return wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
        fields,
    )

def previous_model_directory(model_dir, run_path):
    model_dir = os.path.join(model_dir, run_path)
    
    # Obtain directory
    model_list = os.listdir(model_dir)
    # Save the counter for the max epoch value
    
    max_val = 0
    # Iterate through all the different epochs and obtain the max value
    for i in model_list:
        m = re.search(':(.+?).ckpt',i)
        if m is not None:
            val = int(m.group(1))
            if val > max_val:
                max_val = val
    if f'TestModel:{max_val}.ckpt' in model_list:
        specific_model = f'TestModel:{max_val}.ckpt'
    else:
        specific_model = f'Model:{max_val}.ckpt'
    
    model_dir = os.path.join(model_dir,specific_model)
    return model_dir