import math
from typing import Dict, List, Optional, Tuple

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def find_canonical_etype(g, etype, src_hint=None, dst_hint=None):
    candidates = [ct for ct in g.canonical_etypes if ct[1] == etype]
    if src_hint:
        candidates = [ct for ct in candidates if ct[0] == src_hint]
    if dst_hint:
        candidates = [ct for ct in candidates if ct[2] == dst_hint]
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Edge type '{etype}' is ambiguous: {candidates}")
    return candidates[0]


def _scatter_softmax(scores: th.Tensor, idx: th.Tensor,
                     num_groups: int) -> th.Tensor:
    scores_shifted = scores - scores.max().detach()
    exp_scores     = th.exp(scores_shifted)
    group_sum      = th.zeros(num_groups, dtype=scores.dtype, device=scores.device)
    group_sum.scatter_add_(0, idx, exp_scores)
    return exp_scores / (group_sum[idx] + 1e-9)



class MetaPath2HopAggregator(nn.Module):

    def __init__(self, query_dim: int, feat_dim: int, hidden_dim: int,
                 num_bases: int = 4, dropout: float = 0.0,
                 no_cswd: bool = False):
        super().__init__()
        self.feat_dim  = feat_dim
        self.num_bases = num_bases
        self.no_cswd   = no_cswd  

    
        self.attn_net = nn.Sequential(
            nn.Linear(query_dim + feat_dim, hidden_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        if self.no_cswd:
            
            self.proj_linear = nn.Linear(feat_dim, feat_dim, bias=True)
        
            self.bases        = None
            self.basis_coeffs = None
            self.proj_bias    = None
        else:
           
            self.bases = nn.Parameter(th.Tensor(num_bases, feat_dim, feat_dim))
            nn.init.xavier_uniform_(self.bases.view(num_bases, -1).T)

           
            self.basis_coeffs = nn.Parameter(th.ones(num_bases) / num_bases)

            
            self.proj_bias = nn.Parameter(th.zeros(feat_dim))

            self.proj_linear = None  

        self.dropout = nn.Dropout(dropout)

    def _proj(self, x: th.Tensor) -> th.Tensor:
       
        if self.no_cswd:
            return self.proj_linear(x)                             # (N, feat_dim)
      
        coeffs = th.softmax(self.basis_coeffs, dim=0)             # (num_bases,)
        W = th.einsum('k,kij->ij', coeffs, self.bases)            # (feat_dim, feat_dim)
        return th.matmul(x, W.T) + self.proj_bias                 # (N, feat_dim)

    def forward(
        self,
        g:             dgl.DGLHeteroGraph,
        hop1_cetype:   Tuple[str, str, str],
        hop2_cetype:   Tuple[str, str, str],
        query_embed:   th.Tensor,
        leaf_raw_feat: th.Tensor,
        leaf_mask:     Optional[th.Tensor] = None,
    ) -> th.Tensor:
        
        target_type, etype1, _ = hop1_cetype
        _,           etype2, _ = hop2_cetype

        N_target = g.num_nodes(target_type)
        device   = query_embed.device

        h1_src, h1_dst = g.edges(etype=etype1)
        h2_src, h2_dst = g.edges(etype=etype2)
        E1 = h1_src.shape[0]

        if E1 == 0 or h2_src.shape[0] == 0:
            return th.zeros(N_target, self.feat_dim, device=device)

       
        sort_h2  = th.argsort(h2_src)
        h2_src_s = h2_src[sort_h2]
        h2_dst_s = h2_dst[sort_h2]

        lo     = th.searchsorted(h2_src_s, h1_dst)
        hi     = th.searchsorted(h2_src_s, h1_dst, right=True)
        counts = (hi - lo).clamp(min=0)

        total_2hop = int(counts.sum().item())
        if total_2hop == 0:
            return th.zeros(N_target, self.feat_dim, device=device)

        target_rep   = th.repeat_interleave(h1_src, counts)
        block_lo_rep = lo.repeat_interleave(counts)
        cum_counts   = th.zeros(E1, dtype=th.long, device=device)
        if E1 > 1:
            cum_counts[1:] = counts[:-1].cumsum(0)
        within_block = (th.arange(total_2hop, device=device)
                        - cum_counts.repeat_interleave(counts))
        h2_indices   = block_lo_rep + within_block
        leaf_rep     = h2_dst_s[h2_indices]

    
        if leaf_mask is not None:
            valid_edge = leaf_mask[leaf_rep]  # (total_2hop,) bool
            if not valid_edge.any():
                # 所有邻居都是缺失特征的，返回零向量
                return th.zeros(N_target, self.feat_dim, device=device)
            target_rep = target_rep[valid_edge]
            leaf_rep   = leaf_rep[valid_edge]

       
        q_feats     = query_embed[target_rep]
        l_feats     = leaf_raw_feat[leaf_rep]
        attn_in     = th.cat([q_feats, l_feats], dim=-1)
        attn_scores = self.attn_net(attn_in).squeeze(-1)
        attn_w      = _scatter_softmax(attn_scores, target_rep, N_target)
        attn_w      = self.dropout(attn_w)

        weighted   = attn_w.unsqueeze(-1) * l_feats
        aggregated = th.zeros(N_target, self.feat_dim, device=device)
        aggregated.scatter_add_(
            0,
            target_rep.unsqueeze(-1).expand_as(weighted),
            weighted
        )

      
        return self._proj(aggregated)



class ShapleyPathFusion(nn.Module):
    def __init__(self, num_paths: int, hidden_dim: int, feat_dim: int,
                 exact_threshold: int = 4, ablation_fusion: str = "shapley"):
        super().__init__()
        self.num_paths       = num_paths
        self.hidden_dim      = hidden_dim
        self.feat_dim        = feat_dim
        self.exact_threshold = exact_threshold
        self.ablation_fusion = ablation_fusion

     
        self.value_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
       
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=True)

    def _value(self, feat: th.Tensor) -> th.Tensor:
        return self.value_net(feat).squeeze(-1)

    def _shapley_weights_exact(self,
                                path_feats: List[th.Tensor]) -> th.Tensor:
        K      = len(path_feats)
        N      = path_feats[0].shape[0]
        device = path_feats[0].device
        phi    = th.zeros(N, K, device=device)

        for i in range(K):
            others   = [j for j in range(K) if j != i]
            n_others = K - 1
            for mask in range(1 << n_others):
                subset_others = [others[b] for b in range(n_others)
                                 if mask & (1 << b)]
                s     = len(subset_others)
                coeff = (math.factorial(s) * math.factorial(K - 1 - s)
                         / math.factorial(K))
                paths_with = subset_others + [i]
                feat_with  = th.stack(
                    [path_feats[j] for j in paths_with]).mean(0)
                v_with = self._value(feat_with)
                if subset_others:
                    feat_wo = th.stack(
                        [path_feats[j] for j in subset_others]).mean(0)
                    v_wo = self._value(feat_wo)
                else:
                    v_wo = th.zeros(N, device=device)
                phi[:, i] = phi[:, i] + coeff * (v_with - v_wo)

        return th.softmax(phi, dim=-1)

    def forward(self, path_feats: List[th.Tensor]) -> th.Tensor:
        if self.num_paths == 1:
            return self.out_proj(path_feats[0])

        if self.ablation_fusion == "mean":
            fused = th.stack(path_feats, dim=1).mean(dim=1)
        else:  # shapley
            if self.num_paths <= self.exact_threshold:
                weights = self._shapley_weights_exact(path_feats)
            else:
                indiv   = th.stack([self._value(f) for f in path_feats], dim=1)
                weights = th.softmax(indiv, dim=-1)
            stacked = th.stack(path_feats, dim=1)
            fused   = (weights.unsqueeze(-1) * stacked).sum(dim=1)

        return self.out_proj(fused)



class AttributeCompletionModule(nn.Module):
   

    def __init__(self, completion_configs, feat_dims, query_dim, hidden_dim,
                 num_bases: int = 4,
                 dropout: float = 0.0,
                 ablation_fusion: str = "shapley",
                 no_cswd: bool = False):
        super().__init__()
        self.completion_configs = completion_configs
        self.hidden_dim         = hidden_dim
        self.num_bases          = num_bases
        self.no_cswd            = no_cswd
        self.path_aggregators   = nn.ModuleDict()
        self.path_fusors        = nn.ModuleDict()

        self._self_ref_paths: Dict[str, List[bool]] = {}

        for target_ntype, path_cfgs in completion_configs.items():
            valid_cfgs = [cfg for cfg in path_cfgs
                          if cfg["feat_ntype"] in feat_dims]
            num_paths  = len(valid_cfgs)
            if num_paths == 0:
                continue

            self_ref_flags = []
            for i, cfg in enumerate(valid_cfgs):
                self.path_aggregators[f"{target_ntype}__path{i}"] = \
                    MetaPath2HopAggregator(
                        query_dim=query_dim,
                        feat_dim=feat_dims[cfg["feat_ntype"]],
                        hidden_dim=hidden_dim,
                        num_bases=num_bases,
                        dropout=dropout,
                        no_cswd=no_cswd,
                    )
             
                self_ref_flags.append(cfg["feat_ntype"] == target_ntype)

            self._self_ref_paths[target_ntype] = self_ref_flags

            self.path_fusors[target_ntype] = ShapleyPathFusion(
                num_paths=num_paths,
                hidden_dim=hidden_dim,
                feat_dim=feat_dims[valid_cfgs[0]["feat_ntype"]],
                ablation_fusion=ablation_fusion,
            )

    def shared_parameters(self):
        
        params = []
        for key, agg in self.path_aggregators.items():
            if agg.bases is not None:
                params.append((f"path_aggregators.{key}.bases", agg.bases))
        for key, fusor in self.path_fusors.items():
            for pname, p in fusor.out_proj.named_parameters():
                params.append((f"path_fusors.{key}.out_proj.{pname}", p))
        return params

    def shared_state_dict(self) -> dict:
       
        sd = {}
        for key, agg in self.path_aggregators.items():

            if agg.bases is not None:
                sd[f"path_aggregators.{key}.bases"] = agg.bases.data
        for key, fusor in self.path_fusors.items():
            for pname, p in fusor.out_proj.named_parameters():
                sd[f"path_fusors.{key}.out_proj.{pname}"] = p.data
        return sd

    def load_shared_state_dict(self, shared_sd: dict):
       
        for key, agg in self.path_aggregators.items():
            k = f"path_aggregators.{key}.bases"
            if k in shared_sd and agg.bases is not None:
                agg.bases.data.copy_(shared_sd[k])
        for key, fusor in self.path_fusors.items():
            for pname, p in fusor.out_proj.named_parameters():
                k = f"path_fusors.{key}.out_proj.{pname}"
                if k in shared_sd:
                    p.data.copy_(shared_sd[k])

    def forward(self, g, query_embed_dict,
                raw_feat_dict,
                has_feat_dict: Optional[Dict[str, th.Tensor]] = None,
                ) -> Dict[str, th.Tensor]:
       
        completed: Dict[str, th.Tensor] = {}
        for target_ntype, path_cfgs in self.completion_configs.items():
            if target_ntype not in query_embed_dict:
                continue
            query      = query_embed_dict[target_ntype]
            valid_cfgs = [cfg for cfg in path_cfgs
                          if cfg["feat_ntype"] in raw_feat_dict]
            self_ref_flags = self._self_ref_paths.get(target_ntype, [])
            path_feats = []
            idx        = 0
            for cfg_i, cfg in enumerate(valid_cfgs):
                hop1 = find_canonical_etype(g, cfg["hop1_etype"])
                if hop1 is None: idx += 1; continue
                hop2 = find_canonical_etype(
                    g, cfg["hop2_etype"], src_hint=hop1[2])
                if hop2 is None: idx += 1; continue
                agg_key = f"{target_ntype}__path{idx}"
                if agg_key not in self.path_aggregators: idx += 1; continue

                leaf_mask = None
                if (idx < len(self_ref_flags) and self_ref_flags[idx]
                        and has_feat_dict is not None):
                  
                    feat_ntype = cfg["feat_ntype"]
                    if feat_ntype in has_feat_dict:
                        leaf_mask = has_feat_dict[feat_ntype]

                feat = self.path_aggregators[agg_key](
                    g, hop1_cetype=hop1, hop2_cetype=hop2,
                    query_embed=query,
                    leaf_raw_feat=raw_feat_dict[cfg["feat_ntype"]],
                    leaf_mask=leaf_mask,
                )
                path_feats.append(feat)
                idx += 1

            if path_feats and target_ntype in self.path_fusors:
                fusor = self.path_fusors[target_ntype]
                completed[target_ntype] = (
                    fusor.out_proj(path_feats[0])
                    if len(path_feats) == 1
                    else fusor(path_feats)
                )
        return completed

    def compute_reconstruction_loss(self, g, completed,
                                    device) -> th.Tensor:
        total_loss = th.tensor(0.0, device=device)
        n_terms    = 0

        for target_ntype, completed_feat in completed.items():
            has_gt = g.nodes[target_ntype].data.get("has_gt", None)
            if has_gt is None:
                has_feat = g.nodes[target_ntype].data.get("has_feat", None)
                x_gt     = g.nodes[target_ntype].data.get(
                    "x_gt", g.nodes[target_ntype].data.get("x_full", None))
                if has_feat is None or x_gt is None:
                    continue
                sup_ids = (~has_feat).nonzero(as_tuple=True)[0]
            else:
                x_gt = g.nodes[target_ntype].data.get("x_gt", None)
                if x_gt is None:
                    continue
                sup_ids = has_gt.nonzero(as_tuple=True)[0]

            if len(sup_ids) == 0:
                continue

            pred   = completed_feat[sup_ids]
            target = x_gt[sup_ids].to(device).float()

            pred_n   = F.normalize(pred,   p=2, dim=-1, eps=1e-8)
            target_n = F.normalize(target, p=2, dim=-1, eps=1e-8)
            total_loss = total_loss + F.mse_loss(pred_n, target_n)
            n_terms   += 1

        return total_loss / max(n_terms, 1)