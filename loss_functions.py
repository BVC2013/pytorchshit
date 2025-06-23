import torch
import torch.nn as nn
from sklearn.metrics import silhouette_score

class PopulationLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), lambda_sep=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_sep = lambda_sep

    def forward(self, pred, target):
        base = self.base_loss(pred, target)

        try:
            s_score = silhouette_score(pred.detach().cpu().numpy(), cluster_labels(pred))
            sep_loss = 1 - s_score
        except:
            sep_loss = 0.0

        return base + self.lambda_sep * sep_loss

def cluster_labels(data, k=5):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k).fit_predict(data.detach().cpu().numpy())
