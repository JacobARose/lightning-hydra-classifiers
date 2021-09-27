


import os


# def checkpoints(root): return os.path.join(root, 'checkpoint')
# def logger(root): return os.path.join(root, 'logger')
# def mask(root): return os.path.join(root, 'mask.pth')
# def sparsity_report(root): return os.path.join(root, 'sparsity_report.json')
# def model(root, step): return os.path.join(root, 'model_ep{}_it{}.pth'.format(step.ep, step.it))

def results_history(root): return os.path.join(root, 'results_history.yaml')
def hparams(root): return os.path.join(root, 'hparams.yaml')