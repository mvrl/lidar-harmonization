# useful functions for data metrics/plotting
import numpy as np
import code
import torch
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

from scipy.stats import gaussian_kde

def torch_to_numpy(t):
    if type(t) == torch.Tensor:
        return torch.clone(t).cpu().detach().numpy()
    else:
        return t

def mae(predictions, targets):
    mae = torch.mean(torch.abs(targets - predictions))
    return mae

def create_kde(x, y, xlabel="", ylabel="", output_path=None, sample_size=5000, text=None):
    x = torch_to_numpy(x)
    y = torch_to_numpy(y)
    if sample_size and len(x) > sample_size:
        sample = np.random.choice(len(x), size=sample_size)
        x = x[sample]
        y = y[sample]
    xy = np.vstack([y, x])
    
    z = gaussian_kde(xy)(xy)
    # except np.linalg.LinAlgError:
    #     return

    
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=10)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot([0, 1], [0,1])
    plt.margins(x=0)
    plt.margins(y=0)
    if text:
        plt.text(.5, 0, str(text))
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def create_interpolation_harmonization_plot(
        h_target, h_preds, h_mae, i_target, i_preds, i_mae, phase, output_path):
    
    h_target = torch_to_numpy(h_target)
    h_preds = torch_to_numpy(h_preds)
    i_target = torch_to_numpy(i_target)
    i_preds = torch_to_numpy(i_preds)

    if len(h_target) > 5000:
        sample = np.random.choice(len(h_target), size=5000)
        h_target = h_target[sample]
        h_preds = h_preds[sample]
        i_target = i_target[sample]
        i_preds = i_preds[sample]

    hxy = np.vstack([h_preds, h_target])
    hz = gaussian_kde(hxy)(hxy)
    hidx = hz.argsort()
    h_target, h_preds, hz = h_target[hidx], h_preds[hidx], hz[hidx]

    ixy = np.vstack([i_preds, i_target])
    iz = gaussian_kde(ixy)(ixy)
    iidx = iz.argsort()
    i_target, i_preds, iz = i_target[iidx], i_preds[iidx], iz[iidx]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Results - {phase}", fontsize=16)
    fig.set_size_inches(13, 5)
    ax1.set_title("Harmonization Predictions vs GT")
    ax1.scatter(
            h_target,
            h_preds,
            s=7,
            c=hz)

    ax1.plot([0, 1], [0, 1])
    ax1.margins(x=0)
    ax1.margins(y=0)
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Predictions")

    ax1.text(.5, 0, str(h_mae))

    ax2.set_title("Interpolation Predictions vs GT")
    ax2.scatter(
            i_target,
            i_preds,
            s=7,
            c=iz)

    ax2.text(.5, 0, str(i_mae))

    ax2.plot([0, 1], [0, 1])
    ax2.margins(x=0)
    ax2.margins(y=0)
    ax2.set_xlabel("Ground Truth")
    ax2.set_ylabel("Predictions")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return h_mae, i_mae


    

    
    
def create_bar(flight_count, save_path):
    # flight_count is a dictionary mapping fids to count
    plt.bar(np.arange(len(flight_count.keys())), flight_count.values())
    plt.xticks(np.arange(len(flight_count.keys())), flight_count.keys())
    plt.savefig(save_path)

class Metrics:
    def __init__(self, metrics_list, plots, batch_size):
        self.metrics_list = metrics_list
        self.metrics = {m:None for m in metrics_list}

        # validation metrics
        self.pred_ = None
        self.targ_ = None                
        self.update = True  # some values reset on epoch end

        self.metric_functions = {'mae': mae}

    def collect_metrics(self, predictions, targets):

        if self.update == True:
            self.pred_ = predictions
            self.targ_ = targets
            self.update = False
        else:
            self.pred_ = torch.cat((self.pred_, predictions))
            self.targ_ = torch.cat((self.targ_, targets))
            

    def compute_metrics(self):
        for key in self.metrics:
            self.metrics[key] = self.metric_functions[key](self.pred_, self.targ_)
            
    def clear_metrics(self):
        # clears values - useful for values that reset on epoch end
        self.metrics = {m:None for m in self.metrics_list}
        self.pred_ = None
        self.targ_ = None
        self.update = True

if __name__=='__main__':
    metrics = Metrics(['mae'], None)
    metrics.collect(torch.tensor([2.1, 2., 3.]), torch.tensor([2., 3., .4]))
    metrics.compute()
    print(metrics.metrics)
    metrics.clear_metrics()
    print(metrics.metrics)
    
