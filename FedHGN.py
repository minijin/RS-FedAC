import random
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Optional

import dgl
import torch as th
import torch.nn.functional as F
import tqdm

from Decoders import NodeClassifier, LinkPredictor
from HGNModel import HGNModel
from utils import (get_data_dict, load_data, align_schemas,
                   EarlyStopping,
                   evaluate_recommendation,
                   evaluate_attr_completion)


def get_in_dims(g: dgl.DGLHeteroGraph) -> Optional[dict]:
    in_dims = {}
    for ntype in g.ntypes:
        x = g.nodes[ntype].data.get("x", None)
        if x is not None and x.dtype in (th.float32, th.float64) and x.dim() == 2:
            in_dims[ntype] = x.shape[1]
    return in_dims if in_dims else None


def get_feat_dims(g: dgl.DGLHeteroGraph) -> dict:
  
    feat_dims = {}
    for ntype in g.ntypes:
        x = g.nodes[ntype].data.get("x", None)
        if x is not None and x.dtype in (th.float32, th.float64) and x.dim() == 2:
            feat_dims[ntype] = x.shape[1]
    return feat_dims



def build_rec_dataloader(g: dgl.DGLHeteroGraph, args: Namespace):
    
    user_ntype = args.user_ntype
    train_pos  = g.nodes[user_ntype].data["train_pos_items"]  # (N_user, max_train)

    user_ids = []
    item_ids = []
    for u in range(g.num_nodes(user_ntype)):
        row   = train_pos[u]
        items = row[row >= 0].tolist()
        for i in items:
            user_ids.append(u)
            item_ids.append(i)

    return th.tensor(user_ids, dtype=th.long), th.tensor(item_ids, dtype=th.long)


def bpr_loss(pos_scores: th.Tensor, neg_scores: th.Tensor) -> th.Tensor:

    if neg_scores.dim() == 2:
        pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    else:
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    return loss



class Client:
    def __init__(self, args: Namespace, data: tuple, client_id: int) -> None:
        self.id               = client_id
        self.lr               = args.lr
        self.optim            = args.optim
        self.weight_decay     = args.weight_decay
        self.num_local_epochs = args.num_local_epochs
        self.align_reg        = args.align_reg
        self.ablation         = args.ablation
        self.task             = args.task
        self.device           = args.device
        self.args             = args

        if self.ablation is None:
            self.others_basis_coeffs_encoder = None
            self.others_basis_coeffs_decoder = None

        num_workers = 4 if args.device.type == "cpu" else 0

        g, out_dim, train_nid_dict, val_nid_dict, test_nid_dict = data
        self.g         = g.to(self.device)
        self.ntypes    = g.ntypes
        self.etypes    = list(dict.fromkeys(g.etypes))
        self.canonical_etypes = g.canonical_etypes
        self.out_dim   = out_dim
        self.num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}

        self.train_nid_dict = {k: v.to(self.device) for k, v in train_nid_dict.items()}
        self.val_nid_dict   = {k: v.to(self.device) for k, v in val_nid_dict.items()}
        self.test_nid_dict  = {k: v.to(self.device) for k, v in test_nid_dict.items()}

        in_dims = get_in_dims(g)

       
        completion_configs = getattr(args, 'attr_completion', None)
        feat_dims          = get_feat_dims(g) if completion_configs is not None else None

        if self.task == "node_classification":
            assert len(self.g.ndata["y"].keys()) == 1
            assert len(self.train_nid_dict.keys()) == 1
            self.target_ntype = list(self.train_nid_dict.keys())[0]

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
            self.train_dataloader = dgl.dataloading.DataLoader(
                self.g, self.train_nid_dict, sampler,
                batch_size=args.batch_size, shuffle=True,   drop_last=False,
                num_workers=num_workers, device=args.device, use_uva=False)
            self.val_dataloader = dgl.dataloading.DataLoader(
                self.g, self.val_nid_dict, sampler,
                batch_size=args.batch_size, shuffle=False,  drop_last=False,
                num_workers=num_workers, device=args.device, use_uva=False)
            self.test_dataloader = dgl.dataloading.DataLoader(
                self.g, self.test_nid_dict, sampler,
                batch_size=args.batch_size, shuffle=False,  drop_last=False,
                num_workers=num_workers, device=args.device, use_uva=False)

            self.encoder = HGNModel(
                args, args.hidden_dim, self.ntypes, self.etypes,
                self.canonical_etypes, self.num_nodes_dict,
                in_dims=in_dims,
                completion_configs=completion_configs,
                feat_dims=feat_dims,
            )
            self.encoder.to(args.device)
            self.decoder = NodeClassifier(args.hidden_dim, self.out_dim)
            self.decoder.to(args.device)

        elif self.task == "link_prediction":
            self.user_ntype  = args.user_ntype
            self.item_ntype  = args.item_ntype
            self.inter_etype = args.interaction_etype
            self.n_neg       = args.neg_sample_size

            self.train_pos_users, self.train_pos_items = build_rec_dataloader(g, args)

            self.encoder = HGNModel(
                args, args.hidden_dim, self.ntypes, self.etypes,
                self.canonical_etypes, self.num_nodes_dict,
                in_dims=in_dims,
                completion_configs=completion_configs,
                feat_dims=feat_dims,
            )
            self.encoder.to(args.device)
            self.decoder = LinkPredictor(
                in_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
                scorer=args.get("scorer", "dot") if isinstance(args, dict)
                       else getattr(args, "scorer", "dot"))
            self.decoder.to(args.device)

        else:
            raise ValueError(f"Unknown task: {self.task}")

   
    def local_update(self) -> float:
        if self.optim == "Adam":
            optimizer = th.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim == "SGD":
            optimizer = th.optim.SGD(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim}")

        self.encoder.train()
        self.decoder.train()
        avg_loss = 0

        if self.task == "node_classification":
            avg_loss = self._train_nc(optimizer)
        elif self.task == "link_prediction":
            avg_loss = self._train_lp(optimizer)

        return avg_loss

    def _train_nc(self, optimizer) -> float:
        avg_loss = 0
        with tqdm.tqdm(range(self.num_local_epochs), desc=f"Client {self.id}") as tq:
            for epoch in tq:
                epoch_loss, n_samples = 0, 0
                for _, _, blocks in self.train_dataloader:
                    input_features = get_data_dict(blocks[0].srcdata["x"], blocks[0].srctypes)
                    output_labels  = get_data_dict(blocks[-1].dstdata["y"], blocks[-1].dsttypes)
                    h_dict = self.encoder(blocks, input_features)
                    logits = self.decoder(h_dict[self.target_ntype])
                    loss   = F.nll_loss(F.log_softmax(logits, dim=-1),
                                        output_labels[self.target_ntype])
                    if self.ablation is None:
                        loss = loss + self.compute_alignment_regularization() * self.align_reg
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * logits.shape[0]
                    n_samples  += logits.shape[0]
                epoch_loss /= n_samples
                avg_loss   += epoch_loss
                tq.set_postfix({"loss": f"{epoch_loss:.4f}"}, refresh=False)
        return avg_loss / self.num_local_epochs

    def _train_lp(self, optimizer) -> float:
        
        pos_users       = self.train_pos_users.to(self.device)
        pos_items       = self.train_pos_items.to(self.device)
        n_items         = self.g.num_nodes(self.item_ntype)
        n_pos           = len(pos_users)
        avg_loss        = 0
       
        rec_loss_weight = getattr(self.args, 'rec_loss_weight', None)
        if rec_loss_weight is None:
            rec_loss_weight = 0.1 if getattr(self.args, 'use_attr_completion', False) else 0.0

        with tqdm.tqdm(range(self.num_local_epochs), desc=f"Client {self.id}") as tq:
            for epoch in tq:
                self.encoder.train()
                self.decoder.train()

              
                h_dict     = self.encoder(self.g, {})
                h_user_all = h_dict[self.user_ntype]
                h_item_all = h_dict[self.item_ntype]

            
                perm       = th.randperm(n_pos, device=self.device)
                perm_users = pos_users[perm]
                perm_items = pos_items[perm]

              
                optimizer.zero_grad()

                bpr_total = 0.0
                n_batches = 0
                batch_size = min(2048, n_pos)

                for start in range(0, n_pos, batch_size):
                    end     = min(start + batch_size, n_pos)
                    b_users = perm_users[start:end]
                    b_pos   = perm_items[start:end]
                    B       = len(b_users)

                    b_neg      = th.randint(0, n_items, (B, self.n_neg), device=self.device)
                    h_u        = h_user_all[b_users]
                    h_pos      = h_item_all[b_pos]
                    h_neg      = h_item_all[b_neg]

                    pos_scores = self.decoder(h_u, h_pos)
                    h_u_exp    = h_u.unsqueeze(1).expand(-1, self.n_neg, -1)
                    neg_scores = self.decoder(
                        h_u_exp.reshape(-1, h_u.shape[-1]),
                        h_neg.reshape(-1, h_neg.shape[-1])
                    ).reshape(B, self.n_neg)

                    batch_loss = bpr_loss(pos_scores, neg_scores)
                   
                    (batch_loss * B / n_pos).backward(retain_graph=True)

                    bpr_total += batch_loss.item()
                    n_batches += 1

              
                recon_val = 0.0
                if (rec_loss_weight > 0.0
                        and self.encoder.attr_completion is not None
                        and self.encoder._last_recon_dict):
                    recon_loss = self.encoder.attr_completion.compute_reconstruction_loss(
                        g=self.g,
                        completed=self.encoder._last_recon_dict,
                        device=self.device,
                    )
                   
                    (recon_loss * rec_loss_weight).backward(retain_graph=True)
                    recon_val = recon_loss.item()
                elif rec_loss_weight > 0.0 and epoch == 0:
                    
                    has_recon_module = self.encoder.attr_completion is not None
                    has_recon_dict   = bool(self.encoder._last_recon_dict)
                    n_sup = 0
                    if has_recon_dict:
                        for nt in self.encoder._last_recon_dict:
                            hg = self.g.nodes[nt].data.get("has_gt", None)
                            if hg is not None:
                                n_sup += int(hg.sum().item())
                    print(f"\n[Client {self.id} recon=0 diag] "
                          f"module={has_recon_module}, "
                          f"_last_recon_dict={has_recon_dict}, "
                          f"supervision_nodes={n_sup}")

              
                if self.ablation is None and self.others_basis_coeffs_encoder is not None:
                    align_loss = self.compute_alignment_regularization() * self.align_reg
                    
                    align_loss.backward()

              
                th.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0)
                optimizer.step()

                epoch_bpr   = bpr_total / max(n_batches, 1)
                epoch_total = epoch_bpr + rec_loss_weight * recon_val
                avg_loss   += epoch_total
                tq.set_postfix({
                    "bpr":   f"{epoch_bpr:.4f}",
                    "recon": f"{recon_val:.4f}",
                }, refresh=False)

        return avg_loss / self.num_local_epochs

    
    def local_evaluate(self, is_test: bool = False) -> dict:
        if self.task == "node_classification":
            dataloader = self.test_dataloader if is_test else self.val_dataloader
            return evaluate_node_classification(
                self.encoder, self.decoder, dataloader, self.target_ntype)

        elif self.task == "link_prediction":
           
            rec_metrics = evaluate_recommendation(
                self.encoder, self.decoder, self.g, self.args, is_test=is_test)

            
            comp_metrics = evaluate_attr_completion(
                self.encoder, self.g, self.args)

            return {**rec_metrics, **comp_metrics}

        else:
            raise ValueError(f"Unknown task: {self.task}")

   
    def set_others_basis_coeffs(self, others_basis_coeffs_encoder,
                                others_basis_coeffs_decoder):
        if self.ablation is None:
            self.others_basis_coeffs_encoder = others_basis_coeffs_encoder
            self.others_basis_coeffs_decoder = others_basis_coeffs_decoder
        else:
            raise AssertionError

    def compute_alignment_regularization(self):
        reg = 0
        local_coeffs = th.stack(
            [th.stack(list(pd.values()))
             for pd in self.encoder.basis_coeffs_encoder])
        diff = local_coeffs.unsqueeze(2) - self.others_basis_coeffs_encoder.unsqueeze(1)
        min_diff, _ = th.min(th.sum(th.square(diff), dim=-1), dim=-1)
        reg += min_diff.sum()
        if self.others_basis_coeffs_decoder is not None:
            diff = th.stack(list(self.decoder.basis_coeffs_decoder.values()),
                            dim=0).unsqueeze(1) - self.others_basis_coeffs_decoder
            min_diff, _ = th.min(th.sum(th.square(diff), dim=-1), dim=-1)
            reg += min_diff.sum()
        return reg


class Server:
    def __init__(self, args: Namespace, ntypes, etypes, canonical_etypes,
                 out_dim: Optional[int] = None) -> None:
        self.num_clients = args.num_clients
        self.ablation    = args.ablation

        if self.ablation is None or self.ablation == "B":
            dummy_encoder = HGNModel(
                args, args.hidden_dim,
                ["ntype"], ["etype"], [("ntype", "etype", "ntype")],
                {"ntype": 1}, in_dims=None)
        else:
            dummy_encoder = HGNModel(
                args, args.hidden_dim, ntypes, etypes, canonical_etypes,
                {nt: 1 for nt in ntypes}, in_dims=None)
        dummy_encoder.to(args.device)
        state_dict_encoder = dummy_encoder.state_dict()

        remove = [k for k in state_dict_encoder if k.startswith("embed_layer")]
        remove += [k for k in state_dict_encoder if k.startswith("attr_completion")]
        remove += [k for k in state_dict_encoder if k.startswith("comp_projection")]

        if self.ablation is None or self.ablation == "B":
            remove += [k for k in state_dict_encoder if k.startswith("basis_coeffs_encoder")]
        elif self.ablation == "C":
            remove += [k for k in state_dict_encoder if "bases" in k]
        for k in remove:
            del state_dict_encoder[k]
        self.state_dict_encoder = state_dict_encoder

        # decoder
        if args.task == "node_classification":
            dummy_decoder = NodeClassifier(args.hidden_dim, out_dim)
        elif args.task == "link_prediction":
            dummy_decoder = LinkPredictor(
                in_dim=args.hidden_dim, hidden_dim=args.hidden_dim,
                scorer=getattr(args, "scorer", "dot"))
        else:
            raise ValueError(f"Unknown task: {args.task}")
        dummy_decoder.to(args.device)
        state_dict_decoder = dummy_decoder.state_dict()
        remove = []
        if self.ablation is None or self.ablation == "B":
            remove += [k for k in state_dict_decoder if k.startswith("basis_coeffs_decoder")]
        elif self.ablation == "C":
            remove += [k for k in state_dict_decoder if "bases" in k]
        for k in remove:
            del state_dict_decoder[k]
        self.state_dict_decoder = state_dict_decoder

        if self.ablation is None:
            self.all_clients_basis_coeffs_encoder = [
                th.zeros((args.num_layers, 1, args.num_bases), device=args.device)
                for _ in range(self.num_clients)]
            self.all_clients_basis_coeffs_decoder = None

        
        self.shared_completion_sd = {}

    def send_model(self, client: Client) -> None:
        client.encoder.load_state_dict(self.state_dict_encoder, strict=False)
        client.decoder.load_state_dict(self.state_dict_decoder, strict=False)

        if (client.encoder.attr_completion is not None
                and hasattr(self, 'shared_completion_sd')
                and self.shared_completion_sd):
            client.encoder.attr_completion.load_shared_state_dict(
                self.shared_completion_sd)

        if self.ablation is None:
            others_enc = th.cat(
                [self.all_clients_basis_coeffs_encoder[i]
                 for i in range(self.num_clients) if i != client.id], dim=1)
            others_dec = (None if self.all_clients_basis_coeffs_decoder is None else
                          th.cat([self.all_clients_basis_coeffs_decoder[i]
                                  for i in range(self.num_clients) if i != client.id], dim=0))
            client.set_others_basis_coeffs(others_enc, others_dec)

    def aggregate_model(self, clients, client_weights=None) -> None:
        sd_enc_list    = [c.encoder.state_dict() for c in clients]
        sd_dec_list    = [c.decoder.state_dict() for c in clients]
        client_weights = ([1.0 / len(clients)] * len(clients)
                          if client_weights is None else client_weights)

        for key in self.state_dict_encoder:
            total_w, agg = 0, 0
            for sd, w in zip(sd_enc_list, client_weights):
                if key in sd:
                    agg    += sd[key] * w
                    total_w += w
            self.state_dict_encoder[key] = agg / total_w

        for key in self.state_dict_decoder:
            total_w, agg = 0, 0
            for sd, w in zip(sd_dec_list, client_weights):
                if key in sd:
                    agg    += sd[key] * w
                    total_w += w
            self.state_dict_decoder[key] = agg / total_w

      
        completion_clients = [c for c in clients
                               if c.encoder.attr_completion is not None]
        if completion_clients:
           
            comp_sds = [c.encoder.attr_completion.shared_state_dict()
                        for c in completion_clients]
            comp_ws  = [client_weights[c.id] for c in completion_clients]
            total_w  = sum(comp_ws)
           
            agg_comp_sd = {}
            for key in comp_sds[0]:
                agg_comp_sd[key] = sum(
                    sd[key] * (w / total_w)
                    for sd, w in zip(comp_sds, comp_ws)
                )
            self.shared_completion_sd = agg_comp_sd  

        if self.ablation is None:
            for client in clients:
                self.all_clients_basis_coeffs_encoder[client.id] = th.stack(
                    [th.stack([p.detach() for p in pd.values()])
                     for pd in client.encoder.basis_coeffs_encoder])
                if self.all_clients_basis_coeffs_decoder is not None:
                    self.all_clients_basis_coeffs_decoder[client.id] = th.stack(
                        [p.detach() for p in client.decoder.basis_coeffs_decoder.values()])


class FedHGN:
    def __init__(self, args: Namespace) -> None:
        self.max_rounds   = args.max_rounds
        self.num_clients  = args.num_clients
        self.fraction     = args.fraction
        self.task         = args.task
        self.val_interval = args.val_interval
        self.patience     = args.patience
        self.save_path    = args.save_path
        self.ablation     = args.ablation

        g_list, out_dim, train_nid_dict_list, val_nid_dict_list, test_nid_dict_list = load_data(args)
        ntypes, etypes, canonical_etypes = align_schemas(g_list)

        self.clients = [
            Client(args, (g_list[i], out_dim,
                          train_nid_dict_list[i],
                          val_nid_dict_list[i],
                          test_nid_dict_list[i]), i)
            for i in range(self.num_clients)]
        self.server = Server(args, ntypes, etypes, canonical_etypes, out_dim)

        def w_list(nid_dict_list):
            ws = [sum(len(v) for v in d.values()) for d in nid_dict_list]
            t  = sum(ws)
            return [w / t for w in ws]

        self.train_client_weights = w_list(train_nid_dict_list)
        self.val_client_weights   = w_list(val_nid_dict_list)
        self.test_client_weights  = w_list(test_nid_dict_list)

    def train(self) -> dict:
        
        sample_size = max(round(self.fraction * self.num_clients), 1)
        early_stopping = EarlyStopping(
            patience=self.patience, mode="score",
            save_path=self.save_path, verbose=True)

        monitor_key = "ndcg_at_10"
        best_val_results = {}   

        with tqdm.tqdm(range(self.max_rounds), desc="FedHGN") as tq:
            for round_no in tq:
                selected = random.sample(self.clients, sample_size)
                for c in selected:
                    self.server.send_model(c)

                round_loss = sum(c.local_update() for c in selected) / sample_size

                sel_weights = [self.train_client_weights[c.id] for c in selected]
                total       = sum(sel_weights)
                sel_weights = [w / total for w in sel_weights]
                self.server.aggregate_model(selected, sel_weights)

                tq.set_postfix({"loss": f"{round_loss:.4f}"}, refresh=False)

                if (round_no + 1) % self.val_interval == 0:
                    val_results = self.evaluate(is_test=False)
                    tq.set_postfix(
                        {k: f"{v:.4f}" for k, v in val_results.items()}, refresh=False)

                 
                    _val_snapshot = dict(val_results)  
                    def _save_and_record(path, snap=_val_snapshot):
                        self.save_checkpoint(path)
                        best_val_results.clear()
                        best_val_results.update(snap)

                    early_stopping(val_results[monitor_key],
                                   callback=_save_and_record)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break


        if best_val_results:
            print(f"\n  Best Val @ checkpoint: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in best_val_results.items()))

        return best_val_results

    def evaluate(self, is_test: bool = False) -> dict:
        for c in self.clients:
            self.server.send_model(c)
        weights = self.test_client_weights if is_test else self.val_client_weights

      
        avg:        defaultdict = defaultdict(float)
        comp_total: defaultdict = defaultdict(float)  

        for c, w in zip(self.clients, weights):
            results = c.local_evaluate(is_test)
            for k, v in results.items():
                if k.startswith("comp_"):
                    avg[k]        += v * w
                    comp_total[k] += w
                else:
                    avg[k] += v * w

     
        for k in list(avg.keys()):
            if k.startswith("comp_") and comp_total[k] > 0:
                avg[k] = avg[k] / comp_total[k]

        return dict(avg)

    def save_checkpoint(self, save_path: str) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        th.save(self.server.state_dict_encoder, save_path / "server_encoder.pt")
        th.save(self.server.state_dict_decoder, save_path / "server_decoder.pt")
        if self.ablation is None:
            th.save(self.server.all_clients_basis_coeffs_encoder,
                    save_path / "all_clients_basis_coeffs_encoder.pt")
            th.save(self.server.all_clients_basis_coeffs_decoder,
                    save_path / "all_clients_basis_coeffs_decoder.pt")
        for i, c in enumerate(self.clients):
            th.save(c.encoder.state_dict(), save_path / f"client_{i}_encoder.pt")
            th.save(c.decoder.state_dict(), save_path / f"client_{i}_decoder.pt")

    def load_checkpoint(self, load_path: str) -> None:
        load_path = Path(load_path)
        self.server.state_dict_encoder = th.load(load_path / "server_encoder.pt")
        self.server.state_dict_decoder = th.load(load_path / "server_decoder.pt")
        if self.ablation is None:
            self.server.all_clients_basis_coeffs_encoder = th.load(
                load_path / "all_clients_basis_coeffs_encoder.pt")
            self.server.all_clients_basis_coeffs_decoder = th.load(
                load_path / "all_clients_basis_coeffs_decoder.pt")
        for i, c in enumerate(self.clients):
            c.encoder.load_state_dict(th.load(load_path / f"client_{i}_encoder.pt"))
            c.decoder.load_state_dict(th.load(load_path / f"client_{i}_decoder.pt"))