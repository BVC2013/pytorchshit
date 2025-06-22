import torch
import torch.nn as nn
import torch.nn.functional as F
#early layers learn abstract features
#middle layers are nonlinear transformers
#late layers learn specific features
#Gaussion (GELU) used bc relu is not smooth and can cause issues with optimization
#cons
#might overfit, needs alot alot of data, slo to train
#layers: 256, 512, 1024, 1024, 512, 256
class CompensatorNet(nn.Module):
    def __init__(self, input_dim):
        super(CompensatorNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.3),
            nn.GELU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256, input_dim)  
        )

    def forward(self, x):
        return self.net(x)
