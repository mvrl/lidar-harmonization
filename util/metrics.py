# useful functions for data metrics/plotting
import numpy as np
import code
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from scipy.stats import gaussian_kde

def mae(predictions, targets):
    mae = torch.sum(torch.abs(targets - predictions))/len(predictions)
    return mae

def create_kde(predictions, ground_truths, output_path, sample_ratio=1):
    def torch_to_numpy(t):
        if type(t) == torch.Tensor:
            return torch.clone(t).cpu().detach().numpy()
        else:
            return t

    predictions = torch_to_numpy(predictions)
    ground_truths = torch_to_numpy(ground_truths)    

    xy = np.vstack([predictions, ground_truths])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.scatter(ground_truths, predictions, c=z, s=20, edgecolor='')
    plt.title("Predicted vs Actual")
    plt.ylabel("Predicted")
    plt.xlabel("Ground Truth")
    plt.savefig(output_path)
    

class Metrics:
    def __init__(self, metrics_list, plots):
        self.metrics_list = metrics_list
        self.metrics = {m:None for m in metrics_list}
        # self.plots = {p:None for p in plots}
        self.pred_ = None
        self.targ_ = None
        self.update = -1

        self.metric_functions = {'mae': mae}


    def collect(self, predictions, targets):
        if self.update == -1:
            self.pred_ = predictions
            self.targ_ = targets
        
        self.pred_ = torch.cat((self.pred_, predictions))
        self.targ_ = torch.cat((self.targ_, targets))
        self.update = True

    def compute(self):
        for key in self.metrics:
            self.metrics[key] = self.metric_functions[key](self.pred_, self.targ_)

        self.update = False
            
    def clear_metrics(self):
        # clears values (use every epoch?)
        self.metrics = {m:None for m in self.metrics_list}
        self.pred_ = None
        self.targ_ = None
        self.update = -1

if __name__=='__main__':
    metrics = Metrics(['mae'], None)
    metrics.collect(torch.tensor([2.1, 2., 3.]), torch.tensor([2., 3., .4]))
    metrics.compute()
    print(metrics.metrics)
    metrics.clear_metrics()
    print(metrics.metrics)
        
            
        
        
        
        
    
