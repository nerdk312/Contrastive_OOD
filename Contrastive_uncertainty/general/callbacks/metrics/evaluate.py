import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import faiss

from metric_learning.metrics.Metric_computer import MetricComputer

def evaluate_data(metric_computer,opt,model,dataloader,evaltypes,log_key='Test'):
    computed_metrics, extra_infos = metric_computer.compute_standard(opt, model, dataloader, evaltypes) # Nawid - compute all the metrics

    numeric_metrics = {}
    histogr_metrics = {}
    for main_key in computed_metrics.keys(): # Nawid - iterates through the keys in the computed metrics
        for name,value in computed_metrics[main_key].items(): # Nawid - looks at the name and values of a particular metric
            if isinstance(value, np.ndarray):
                if main_key not in histogr_metrics: histogr_metrics[main_key] = {} # Nawid - add empty dict for the situation where the main key is not in the histogram metrics
                histogr_metrics[main_key][name] = value
            else:
                if main_key not in numeric_metrics: numeric_metrics[main_key] = {}
                numeric_metrics[main_key][name] = value

  ###
    full_result_str = ''
    for evaltype in numeric_metrics.keys(): # Nawid - Go through the different types
        full_result_str += 'Embed-Type: {}:\n'.format(evaltype) # Shows the embed type
        # Nawid - Shows all the different results
        for i,(metricname, metricval) in enumerate(numeric_metrics[evaltype].items()): # Nawid - Go through the different metric
            full_result_str += '{0}{1}: {2:4.4f}'.format(' | ' if i>0 else '',metricname, metricval)
        full_result_str += '\n'

    print(full_result_str)
    # Nawid - log the histograms
    if True:
        for evaltype in histogr_metrics.keys():
            for eval_metric, hist in histogr_metrics[evaltype].items(): # Nawid - plot the histogram metrics on wandb
                import wandb, numpy
                wandb.log({log_key+': '+evaltype+'_{}'.format(eval_metric): wandb.Histogram(np_histogram=(list(hist),list(np.arange(len(hist)+1))))}, step=1)
                wandb.log({log_key+': '+evaltype+'_LOG-{}'.format(eval_metric): wandb.Histogram(np_histogram=(list(np.log(hist)+20),list(np.arange(len(hist)+1))))}, step=1)

    for evaltype in numeric_metrics.keys():# Nawid - plot the numeric metrics on wandb
        for eval_metric in numeric_metrics[evaltype].keys():
            parent_metric = evaltype+'_{}'.format(eval_metric.split('@')[0])
            wandb.run.summary[eval_metric] = numeric_metrics[evaltype][eval_metric]
            #wandb.log({eval_metric:numeric_metrics[evaltype][eval_metric]})
            #print('parent metric',parent_metric)
