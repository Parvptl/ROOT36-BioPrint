import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=16, embedding_dim=16, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward_one(self, x):
        # We can explicitly handle NaNs by replacing them with 0 (since we might normalize the inputs)
        x = torch.nan_to_num(x, nan=0.0)
        return self.encoder(x)
        
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2
        
    def get_embedding(self, x):
        return self.forward_one(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0, metric="euclidean"):
        super().__init__()
        self.margin = margin
        self.metric = metric
        
    def forward(self, out1, out2, label):
        # label: 1 if same user, 0 if different user
        if self.metric == "euclidean":
            euclidean_distance = F.pairwise_distance(out1, out2, keepdim=True)
            loss_contrastive = torch.mean(
                label * torch.pow(euclidean_distance, 2) +
                (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            )
        elif self.metric == "cosine":
            # For cosine, distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(out1, out2).unsqueeze(1)
            dist = 1 - cos_sim
            loss_contrastive = torch.mean(
                label * torch.pow(dist, 2) +
                (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
            )
        else:
            raise ValueError(f"Unknown metric {self.metric}")
            
        return loss_contrastive
