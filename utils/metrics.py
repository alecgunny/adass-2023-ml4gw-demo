import torch
from torchmetrics import Metric


class RecoveredRecall(Metric):
    def __init__(self, fpr = 0.0):
        self.fpr = fpr
        self.add_state("y_hat", default=[])
        self.add_state("y", default=[])

    def update(self, y_hat, y):
        self.y_hat.append(y_hat)
        self.y.append(y)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)

        k = int(y_hat.size(0) * self.fpr)
        
        