"""
lightning_hydra_classifiers/utils/metric_utils.py

Author: Jacob A Rose
Created: Thursday Junee 11th, 2021

"""



import torchmetrics as metrics
# from typing import List





def get_scalar_metrics(num_classes: int,
                       average: str='macro', 
                       prefix: str=''
                      ) -> metrics.MetricCollection:
    default = {'acc_top1': metrics.Accuracy(top_k=1),
               'acc_top3': metrics.Accuracy(top_k=3),
               'precision_top1': metrics.Precision(num_classes=num_classes, top_k=1, average=average),
               'recall_top1': metrics.Recall(num_classes=num_classes, top_k=1, average=average)}
    if len(prefix)>0:
        for k in list(default.keys()):
            default[prefix + r'/' + k] = default[k]
            del default[k]
    
    return metrics.MetricCollection(default)


def get_per_class_metrics(num_classes: int,
                          prefix: str=''
                         ) -> metrics.MetricCollection:
    
    default = {'F1': metrics.F1(num_classes=num_classes, average=None)}
    
    if len(prefix)>0:
        for k in list(default.keys()):
            default[prefix + r'/' + k] = default[k]
            del default[k]
    
    return metrics.MetricCollection(default)