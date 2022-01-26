import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from PIL import Image
import faiss

import numpy as np
import faiss
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
import copy

from Contrastive_uncertainty.general.callbacks.metrics import e_recall, nmi, f1, mAP, mAP_c, mAP_1000, mAP_lim
from Contrastive_uncertainty.general.callbacks.metrics import dists, rho_spectrum, uniformity
from Contrastive_uncertainty.general.callbacks.metrics import c_recall, c_nmi, c_f1, c_mAP_c, c_mAP_1000, c_mAP_lim
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading



def select(metricname, pl_module):
    #### Metrics based on euclidean distances
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)
    elif metricname=='nmi':
        return nmi.Metric()
    elif metricname=='mAP':
        return mAP.Metric()
    elif metricname=='mAP_c':
        return mAP_c.Metric()
    elif metricname=='mAP_lim':
        return mAP_lim.Metric()
    elif metricname=='mAP_1000':
        return mAP_1000.Metric()
    elif metricname=='f1':
        return f1.Metric()

    #### Metrics based on cosine similarity
    elif 'c_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return c_recall.Metric(k)
    elif metricname=='c_nmi':
        return c_nmi.Metric()
    elif metricname=='c_mAP':
        return c_mAP.Metric()
    elif metricname=='c_mAP_c':
        return c_mAP_c.Metric()
    elif metricname=='c_mAP_lim':
        return c_mAP_lim.Metric()
    elif metricname=='c_mAP_1000':
        return c_mAP_1000.Metric()
    elif metricname=='c_f1':
        return c_f1.Metric()

    #### Generic Embedding space metrics
    elif 'dists' in metricname:
        mode = metricname.split('@')[-1]
        return dists.Metric(mode)
    elif 'rho_spectrum' in metricname:
        mode = int(metricname.split('@')[-1])
        embed_dim = pl_module.hparams.emb_dim
        return rho_spectrum.Metric(embed_dim, mode=mode)
    elif 'uniformity' in metricname:
        t = 2
        return uniformity.Metric(t=t)
    else:
        raise NotImplementedError("Metric {} not available!".format(metricname))


class MetricLogger(pl.Callback):
    def __init__(self, metric_names,datamodule,evaltypes,
        vector_level: str='instance',
        label_level:str='fine',
        quick_callback:bool=True):

        super().__init__()
        self.metric_names = metric_names # Nawid - names of the metrics to compute
        
        self.vector_level = vector_level
        self.label_level = label_level

        self.datamodule = datamodule
        self.num_fine_classes = self.datamodule.num_classes
        self.num_coarse_classes = self.datamodule.num_coarse_classes
        self.dataloader = self.datamodule.val_dataloader()
        # Separate the val loader into two separate parts
        
        if len(self.dataloader)> 1:
            _, self.dataloader = self.dataloader
        #import ipdb; ipdb.set_trace()
        self.evaltypes = evaltypes
        self.quick_callback = quick_callback
        
    def metric_initialise(self, trainer,pl_module):
        self.list_of_metrics = [select(metricname, pl_module) for metricname in self.metric_names] # Nawid - Obtains the different metrics for the task
        self.requires        = [metric.requires for metric in self.list_of_metrics] # Nawid - says what each metric requires
        self.requires        = list(set([x for y in self.requires for x in y]))

    def compute_standard(self, trainer, pl_module):

        self.vector_dict = {'vector_level':{'instance':pl_module.instance_vector, 'fine':pl_module.fine_vector, 'coarse':pl_module.coarse_vector},
        'label_level':{'fine':0,'coarse':1},
        'num_classes':{'fine':self.num_fine_classes,'coarse':self.num_coarse_classes}} 
        

        evaltypes = copy.deepcopy(self.evaltypes)
        n_classes = self.vector_dict['num_classes'][self.label_level]

        pl_module.to(pl_module.device).eval()

        ###
        feature_colls  = {key:[] for key in evaltypes}

        ###
        with torch.no_grad():
            target_labels = []
            loader = quickloading(self.quick_callback,self.dataloader) # Used to choose using a single dataloader or using a wide variety of data loaders
            assert len(loader) >0, 'loader is empty'
            final_iter = tqdm(loader, desc='Embedding Data...'.format(len(evaltypes)))# Nawid - loading of dataloader I believe
            
            for i, (images, *labels, indices) in enumerate(final_iter): # Obtain data and labels from dataloader
                if isinstance(images, tuple) or isinstance(images, list):
                    images, *aug_imgs = images

                # Selects the correct label based on the desired label level
                if len(labels) > 1:
                    label_index = self.vector_dict['label_level'][self.label_level]
                    labels = labels[label_index]
                else: # Used for the case of the OOD data
                    labels = labels[0]

                target_labels.extend(labels.numpy().tolist())  # Nawid- obtain labels
                out = self.vector_dict['vector_level'][self.vector_level](images.to(pl_module.device))
                if isinstance(out, tuple): out, *aux_f = out #  Nawid - if the output is a tuple, separate the output

                ### Include embeddings of all output features
                for evaltype in evaltypes:
                    if isinstance(out, dict): # Nawid - if out is a dict
                        feature_colls[evaltype].extend(out[evaltype].cpu().detach().numpy().tolist()) # Nawid - obtain the part of out required if it is a dict
                    else:
                        feature_colls[evaltype].extend(out.cpu().detach().numpy().tolist()) # Nawid - add out

            target_labels = np.hstack(target_labels).reshape(-1,1) # Nawid- reshape labels


        computed_metrics = {evaltype:{} for evaltype in evaltypes} # Nawid - dict for logging metric
        extra_infos      = {evaltype:{} for evaltype in evaltypes} # Nawid - dict for extra info


        ###
        faiss.omp_set_num_threads(6) # Set 6 workers
        # faiss.omp_set_num_threads(self.pars.kernels)
        res = None
        torch.cuda.empty_cache()
        if pl_module.device == 'cuda': res = faiss.StandardGpuResources()

        import time
        for evaltype in evaltypes:
            #print('feature colls',feature_colls[evaltype])
            features        = np.vstack(feature_colls[evaltype]).astype('float32') # Nawid - stack features
            features_cosine = normalize(features, axis=1) #  Nawid - normalise features for cosine

            start = time.time()
            # Nawid - compute different properties which may be required for the metrics
            """============ Compute k-Means ==============="""
            if 'kmeans' in self.requires:
                ### Set CPU Cluster index
                cluster_idx = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans            = faiss.Clustering(features.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                ### Train Kmeans
                kmeans.train(features, cluster_idx)
                centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])

            if 'kmeans_cosine' in self.requires:
                ### Set CPU Cluster index
                cluster_idx = faiss.IndexFlatL2(features_cosine.shape[-1])
                if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans            = faiss.Clustering(features_cosine.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                ### Train Kmeans
                kmeans.train(features_cosine, cluster_idx)
                centroids_cosine = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features_cosine.shape[-1])
                centroids_cosine = normalize(centroids,axis=1)


            """============ Compute Cluster Labels ==============="""
            if 'kmeans_nearest' in self.requires:
                faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids)
                _, computed_cluster_labels = faiss_search_index.search(features, 1)

            if 'kmeans_nearest_cosine' in self.requires:
                faiss_search_index = faiss.IndexFlatIP(centroids_cosine.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids_cosine)
                _, computed_cluster_labels_cosine = faiss_search_index.search(features_cosine, 1)



            """============ Compute Nearest Neighbours ==============="""
            if 'nearest_features' in self.requires:
                faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(features)

                max_kval            = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
                _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))

                k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

            if 'nearest_features_cosine' in self.requires:
                faiss_search_index  = faiss.IndexFlatIP(features_cosine.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(normalize(features_cosine,axis=1))

                max_kval                   = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
                _, k_closest_points_cosine = faiss_search_index.search(normalize(features_cosine,axis=1), int(max_kval+1))
                k_closest_classes_cosine   = target_labels.reshape(-1)[k_closest_points_cosine[:,1:]]


            ###
            if pl_module.device=='cuda': # Nawid - Place features on GPU
                features        = torch.from_numpy(features).to(self.pars['device'])
                features_cosine = torch.from_numpy(features_cosine).to(self.pars['device'])

            start = time.time()
            for metric in self.list_of_metrics: #  Nawid - go through the metric one by one, place required information in the dict and then compute the metric
                input_dict = {}
                if 'features' in metric.requires:         input_dict['features'] = features
                if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels

                if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
                if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
                if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes

                if 'features_cosine' in metric.requires:         input_dict['features_cosine'] = features_cosine

                if 'kmeans_cosine' in metric.requires:           input_dict['centroids_cosine'] = centroids_cosine
                if 'kmeans_nearest_cosine' in metric.requires:   input_dict['computed_cluster_labels_cosine'] = computed_cluster_labels_cosine
                if 'nearest_features_cosine' in metric.requires: input_dict['k_closest_classes_cosine'] = k_closest_classes_cosine

                computed_metrics[evaltype][metric.name] = metric(**input_dict) #  Nawid - Use the required inforamtion to compute a particular metric

            extra_infos[evaltype] = {'features':features, 'target_labels':target_labels}
        torch.cuda.empty_cache()
        return computed_metrics, extra_infos



    def evaluate_data(self,trainer,pl_module,log_key='Test'):
        computed_metrics, extra_infos = self.compute_standard(trainer,pl_module) # Nawid - compute all the metrics

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

        # Nawid - log the histograms
        # if log online
        for evaltype in histogr_metrics.keys():
            for eval_metric, hist in histogr_metrics[evaltype].items(): # Nawid - plot the histogram metrics on wandb
                import wandb, numpy
                               
                wandb.log({f'{log_key}: {evaltype}: {eval_metric}: {self.vector_level}: {self.label_level}': wandb.Histogram(np_histogram=(list(hist),list(np.arange(len(hist)+1))))}, step=trainer.global_step)
                wandb.log({f'{log_key}: {evaltype} LOG-{eval_metric}:{self.vector_level}: {self.label_level}': wandb.Histogram(np_histogram=(list(np.log(hist)+20),list(np.arange(len(hist)+1))))}, step=trainer.global_step)
                #wandb.log({log_key+': '+evaltype+'_{}'.format(eval_metric): wandb.Histogram(np_histogram=(list(hist),list(np.arange(len(hist)+1))))}, step=trainer.global_step)
                #wandb.log({log_key+': '+evaltype+'_LOG-{}'.format(eval_metric): wandb.Histogram(np_histogram=(list(np.log(hist)+20),list(np.arange(len(hist)+1))))}, step=trainer.global_step)

        for evaltype in numeric_metrics.keys():# Nawid - plot the numeric metrics on wandb
            for eval_metric in numeric_metrics[evaltype].keys():
                parent_metric = evaltype+'_{}'.format(eval_metric.split('@')[0])
                if 'dists' in eval_metric or 'rho_spectrum' in eval_metric:
                    wandb.log({f'{eval_metric}: {self.vector_level}: {self.label_level}':numeric_metrics[evaltype][eval_metric]})
                else:
                    wandb.run.summary[f'{eval_metric}: {self.vector_level}: {self.label_level}'] = numeric_metrics[evaltype][eval_metric]
                #wandb.log({eval_metric:numeric_metrics[evaltype][eval_metric]})
                #print('parent metric',parent_metric)

    def on_validation_epoch_end(self,trainer,pl_module):
        self.metric_initialise(trainer,pl_module)
        self.evaluate_data(trainer,pl_module)
    '''
    def on_test_epoch_end(self,trainer,pl_module):
        self.metric_initialise(trainer,pl_module)
        self.evaluate_data(trainer,pl_module)
    '''

evaluation_metrics =['e_recall@1', 'e_recall@2', 'e_recall@4', 'nmi', 'f1', 'mAP_1000', 'mAP_lim', 'mAP_c', \
                        'dists@intra', 'dists@inter', 'dists@intra_over_inter', 'rho_spectrum@0', \
                        'rho_spectrum@-1', 'rho_spectrum@1', 'rho_spectrum@2', 'rho_spectrum@10','uniformity']

evaltypes = ['discriminative']