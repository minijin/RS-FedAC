import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NodeClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pre_activation=F.relu) -> None:
        super(NodeClassifier, self).__init__()
        self.pre_activation = pre_activation
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: th.FloatTensor) -> th.FloatTensor:
        return self.linear(self.pre_activation(x)) if self.pre_activation else self.linear(x)


class LinkPredictor(nn.Module):
    
    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 scorer: str = "dot",
                 temperature: float = 1.0) -> None:
        super(LinkPredictor, self).__init__()
        self.scorer      = scorer
        self.temperature = temperature  

        if scorer == "dot":
            pass

        elif scorer == "bilinear":
            self.W = nn.Parameter(th.Tensor(in_dim, in_dim))
            nn.init.xavier_uniform_(self.W)

        elif scorer == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown scorer: {scorer}. Choose from dot/bilinear/mlp")

    def forward(self, h_user: th.FloatTensor,
                h_item: th.FloatTensor) -> th.FloatTensor:
     
        if self.scorer == "dot":
        
            if h_item.dim() == 3:
                scores = th.bmm(h_item, h_user.unsqueeze(-1)).squeeze(-1)  # (B, N)
            else:
                scores = (h_user * h_item).sum(dim=-1)                     # (B,)
            return scores / self.temperature

        elif self.scorer == "bilinear":
            if h_item.dim() == 3:
                h_user_proj = th.matmul(h_user, self.W)
                return th.bmm(h_item, h_user_proj.unsqueeze(-1)).squeeze(-1)
            else:
                h_user_proj = th.matmul(h_user, self.W)
                return (h_user_proj * h_item).sum(dim=-1)

        else:  # mlp
            if h_item.dim() == 3:
                B, N, D = h_item.shape
                h_user_exp = h_user.unsqueeze(1).expand(B, N, D)
                combined   = th.cat([h_user_exp, h_item], dim=-1)
                return self.mlp(combined).squeeze(-1)
            else:
                combined = th.cat([h_user, h_item], dim=-1)
                return self.mlp(combined).squeeze(-1)