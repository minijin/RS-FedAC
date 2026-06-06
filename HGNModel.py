from argparse import Namespace
from collections.abc import Callable
from itertools import chain
from typing import Optional, Union, Dict

import dgl
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils import get_data_dict
from AttributeCompletion import AttributeCompletionModule


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, etypes, num_bases, *, use_weight=True,
                 use_bias=True, activation=None, use_self_loop=False, dropout=0.0):
        super().__init__()
        self.in_dim=in_dim; self.out_dim=out_dim; self.etypes=etypes
        self.num_bases=num_bases; self.use_weight=use_weight; self.use_bias=use_bias
        self.activation=activation; self.use_self_loop=use_self_loop
        self.conv=dglnn.HeteroGraphConv(
            {e:dglnn.GraphConv(in_dim,out_dim,norm='right',weight=False,bias=False) for e in etypes})
        if use_weight:
            self.bases=nn.Parameter(th.Tensor(num_bases,in_dim,out_dim))
            nn.init.xavier_uniform_(self.bases,gain=nn.init.calculate_gain('relu'))
        if use_bias:
            self.h_bias=nn.Parameter(th.Tensor(out_dim)); nn.init.zeros_(self.h_bias)
        if use_self_loop:
            self.loop_weight=nn.Parameter(th.Tensor(in_dim,out_dim))
            nn.init.xavier_uniform_(self.loop_weight,gain=nn.init.calculate_gain('relu'))
        self.dropout=nn.Dropout(dropout)

    def forward(self, g, inputs, basis_coeffs):
        with g.local_scope():
            w_dict={}
            if self.use_weight:
                for e in self.etypes:
                    w_dict[e]={"weight":th.matmul(basis_coeffs[e],self.bases.view(self.num_bases,-1)).view(self.in_dim,self.out_dim)}
            inputs_dst=({k:v[:g.number_of_dst_nodes(k)] for k,v in inputs.items()} if g.is_block else inputs)
            hs=self.conv(g,inputs,mod_kwargs=w_dict)
            def _apply(nt,h):
                if self.use_self_loop: h=h+th.matmul(inputs_dst[nt],self.loop_weight)
                if self.use_bias:     h=h+self.h_bias
                if self.activation:   h=self.activation(h)
                return self.dropout(h)
            return {nt:_apply(nt,h) for nt,h in hs.items()}


class RGCN(nn.Module):
    def __init__(self, hidden_dim, out_dim, etypes, num_bases, *,
                 num_hidden_layers=1, dropout=0.0, use_self_loop=False):
        super().__init__()
        self.layers=nn.ModuleList()
        self.layers.append(RGCNLayer(hidden_dim,hidden_dim,etypes,num_bases,
                                     activation=F.relu,use_self_loop=use_self_loop,dropout=dropout,use_weight=False))
        for _ in range(num_hidden_layers):
            self.layers.append(RGCNLayer(hidden_dim,hidden_dim,etypes,num_bases,
                                         activation=F.relu,use_self_loop=use_self_loop,dropout=dropout))
        self.layers.append(RGCNLayer(hidden_dim,out_dim,etypes,num_bases,
                                     activation=None,use_self_loop=use_self_loop))

    def forward(self, g, inputs, basis_coeffs_encoder):
        h=inputs
        if isinstance(g,dgl.DGLHeteroGraph):
            for layer,bc in zip(self.layers,chain([None],basis_coeffs_encoder)):
                h=layer(g,h,bc)
        else:
            for layer,block,bc in zip(self.layers,g,chain([None],basis_coeffs_encoder)):
                h=layer(block,h,bc)
        return h


class MixedEmbedLayer(nn.Module):
    def __init__(self, in_dims, no_feat_ntypes, hidden_dim):
        super().__init__()
        self.linear_layers=nn.ModuleDict({nt:nn.Linear(d,hidden_dim) for nt,d in in_dims.items()})
        self.embed_layers =nn.ModuleDict({nt:nn.Embedding(n,hidden_dim) for nt,n in no_feat_ntypes.items()})
        self.feat_ntypes   =set(in_dims.keys())
        self.no_feat_ntypes=set(no_feat_ntypes.keys())

    def forward(self, feat_dict, nid_dict):
        h={}
        for nt,feat in feat_dict.items(): h[nt]=self.linear_layers[nt](feat.float())
        for nt,nids in nid_dict.items():  h[nt]=self.embed_layers[nt](nids)
        return h


class HGNModel(nn.Module):

    def __init__(self, args, out_dim, ntypes, etypes, canonical_etypes,
                 num_nodes_dict, in_dims=None, completion_configs=None, feat_dims=None):
        super().__init__()
        self.model_name      =args.model
        self.num_bases       =args.num_bases
        self.ntypes          =ntypes
        self.etypes          =etypes
        self.canonical_etypes=canonical_etypes

        if in_dims is None:
            self.embed_mode ="id"
            self.embed_layer=dglnn.HeteroEmbedding(num_nodes_dict,args.hidden_dim)
        else:
            feat_nt   ={nt:d for nt,d in in_dims.items() if nt in ntypes}
            no_feat_nt={nt:num_nodes_dict[nt] for nt in ntypes if nt not in in_dims}
            if len(no_feat_nt)==0:
                self.embed_mode ="feat"
                self.embed_layer=dglnn.HeteroLinear(feat_nt,args.hidden_dim)
            else:
                self.embed_mode ="mixed"
                self.embed_layer=MixedEmbedLayer(feat_nt,no_feat_nt,args.hidden_dim)

        if (completion_configs is not None and feat_dims is not None
                and getattr(args,'use_attr_completion',False)):

            num_bases_comp = min(4, getattr(args, 'num_bases', 4))
            self.attr_completion=AttributeCompletionModule(
                completion_configs=completion_configs, feat_dims=feat_dims,
                query_dim=args.hidden_dim, hidden_dim=args.hidden_dim,
                num_bases=num_bases_comp,
                dropout=getattr(args,'dropout',0.0),
                ablation_fusion=getattr(args,'ablation_fusion','shapley'),
                no_cswd=getattr(args,'no_completion_cswd', False),
            )

    
            self.comp_projection = nn.ModuleDict()
            for nt in completion_configs.keys():
                if nt in feat_dims:
                    self.comp_projection[nt] = nn.Linear(
                        feat_dims[nt], args.hidden_dim)
        else:
            self.attr_completion=None
            self.comp_projection=None

        self._last_recon_dict: Dict[str,th.Tensor]={}

    
        assert args.num_layers>1
        self.basis_coeffs_encoder=nn.ModuleList()
        for _ in range(args.num_layers-1):
            pd=nn.ParameterDict()
            for e in self.etypes:
                pd[e]=nn.Parameter(th.Tensor(self.num_bases))
                nn.init.xavier_uniform_(pd[e].view(1,-1),gain=nn.init.calculate_gain('relu'))
            self.basis_coeffs_encoder.append(pd)
        self.model=RGCN(args.hidden_dim,out_dim,etypes,self.num_bases,
                        num_hidden_layers=args.num_layers-2,
                        dropout=args.dropout,use_self_loop=args.use_self_loop)

    def _get_linear_for_ntype(self, ntype: str) -> Optional[nn.Linear]:
    
        if self.embed_mode == "feat":
            linears = getattr(self.embed_layer, "linears", None)
            if linears is not None and ntype in linears:
                return linears[ntype]
            return None
        elif self.embed_mode == "mixed":
            ll = self.embed_layer.linear_layers
            return ll[ntype] if ntype in ll else None
        return None  

    def _get_initial_embeddings(self, g):
        if self.embed_mode == "id":
            nids = ({nt:g.nodes(nt) for nt in g.ntypes} if isinstance(g,dgl.DGLHeteroGraph)
                    else get_data_dict(g[0].srcdata[dgl.NID],g[0].srctypes))
            h_dict = self.embed_layer(nids)

        elif self.embed_mode == "feat":
            if isinstance(g,dgl.DGLHeteroGraph):
                feat_dict={nt:g.nodes[nt].data['x'] for nt in g.ntypes}
            else:
                src_x=g[0].srcdata['x']
                feat_dict=({nt:src_x[nt] for nt in g[0].srctypes} if isinstance(src_x,dict)
                           else {g[0].srctypes[0]:src_x})
            h_dict = self.embed_layer(feat_dict)

        else:  
            fn=self.embed_layer.feat_ntypes; nfn=self.embed_layer.no_feat_ntypes
            if isinstance(g,dgl.DGLHeteroGraph):
                feat_dict={nt:g.nodes[nt].data['x'] for nt in g.ntypes if nt in fn}
                nid_dict ={nt:g.nodes(nt)           for nt in g.ntypes if nt in nfn}
            else:
                b0=g[0]; src_x=b0.srcdata.get('x',{}); src_nid=b0.srcdata.get(dgl.NID,{})
                feat_dict={nt:(src_x[nt] if isinstance(src_x,dict) else src_x)
                           for nt in b0.srctypes if nt in fn}
                nid_dict ={nt:(src_nid[nt] if isinstance(src_nid,dict) else src_nid)
                           for nt in b0.srctypes if nt in nfn}
            h_dict = self.embed_layer(feat_dict,nid_dict)

        self._last_recon_dict = {}

        if self.attr_completion is not None and isinstance(g,dgl.DGLHeteroGraph):
            raw_feat_dict = {nt:g.nodes[nt].data['x']
                             for nt in g.ntypes
                             if g.nodes[nt].data.get('x') is not None
                             and g.nodes[nt].data['x'].dtype in (th.float32,th.float64)
                             and g.nodes[nt].data['x'].dim()==2}
            query_dict = {k: v.detach() for k, v in h_dict.items()}

            completed = self.attr_completion(g, query_dict, raw_feat_dict)
            self._last_recon_dict = completed

            for ntype, comp_feat in completed.items():
                
                if self.comp_projection is None or ntype not in self.comp_projection:
                    continue
                proj = self.comp_projection[ntype]

                h_comp = proj(comp_feat.float())  # (N, hidden_dim)

                has_feat = g.nodes[ntype].data.get("has_feat", None)

                if has_feat is None
                    h_dict[ntype] = h_comp

                else:
                    mask_2d = has_feat.to(comp_feat.device).unsqueeze(-1)
                    h_dict[ntype] = th.where(mask_2d, h_dict[ntype], h_comp)

        return h_dict

    def forward(self, g, inputs):
        h_dict = self._get_initial_embeddings(g)
        return self.model(g, h_dict, self.basis_coeffs_encoder)