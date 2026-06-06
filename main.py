import argparse

import torch as th

from FedHGN import FedHGN
from utils import load_configs, get_save_path, set_random_seeds, print_results, save_results


def main(args):
    if args.framework == "FedHGN":
        fl_framework = FedHGN(args)
    else:
        raise ValueError("Unknown framework.")

   
    best_val_results = fl_framework.train()

    
    fl_framework.load_checkpoint(args.save_path)
    test_results = fl_framework.evaluate(is_test=True)

    
    print(f"\n{'='*60}")
    print("Best Validation Results (saved checkpoint):")
    if best_val_results:
        print_results(best_val_results)
    else:
        print("  (not available)")


    save_results(test_results, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FedHGN')
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--split-strategy", "-s", type=str, default="edges")
    parser.add_argument("--framework", "-f", type=str, default="FedHGN")
    parser.add_argument("--ablation", "-a", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="RGCN")
    parser.add_argument("--num-clients", "-c", type=int, default=3)
    parser.add_argument("--gpu", '-g', type=int, default=-1)
    parser.add_argument("--random-seed", type=int, default=1000)
    parser.add_argument("--config-path", type=str, default="./configs.yaml")

   
    parser.add_argument("--use-attr-completion", action="store_true")
    parser.add_argument(
        "--ablation-fusion", type=str, default="shapley",
        choices=["shapley", "mean"])

    parser.add_argument(
        "--no-completion-cswd", action="store_true",
        help=(
            "Ablation: disable C-SWD in the completion module. "
            "bases in MetaPath2HopAggregator become fully client-local "
            "and are NOT uploaded to / downloaded from the server."
        ),
    )

    parser.add_argument(
        "--rec-loss-weight", type=float, default=None,
        help=(
            "Weight for reconstruction loss λ_rec. "
            "If not set, falls back to rec_loss_weight in configs.yaml "
            "(RGCN_lp_feat_Yelp), or 0.1 when --use-attr-completion is on."
        ),
    )


    parser.add_argument(
        "--temperature", type=float, default=None,
        help=(
            "Temperature τ for score scaling: s(u,v) = h_u·h_v / τ. "
            "Falls back to 'temperature' in configs.yaml, then 1.0."
        ),
    )

    args = parser.parse_args()
    args = load_configs(args)

    if args.temperature is not None:
        pass  
    elif not hasattr(args, 'temperature') or args.temperature is None:
        args.temperature = 1.0  

  
    if args.rec_loss_weight is not None:
        pass
    elif not hasattr(args, 'rec_loss_weight') or args.rec_loss_weight is None:
        if args.use_attr_completion:
            args.rec_loss_weight = 0.1   
        else:
            args.rec_loss_weight = 0.0   

    if args.gpu >= 0 and th.cuda.is_available():
        args.device = th.device(f"cuda:{args.gpu}")
    else:
        args.device = th.device("cpu")
    args.save_path = get_save_path(args)

    set_random_seeds(args.random_seed)

    print(f"\n{'='*60}")
    print(f"Dataset:         {args.dataset}")
    print(f"Framework:       {args.framework}")
    print(f"Task:            {args.task}")
    print(f"Split:           {args.split_strategy}")
    print(f"Clients:         {args.num_clients}")
    print(f"Device:          {args.device}")
    print(f"AttrCompletion:  {args.use_attr_completion}")
    if args.use_attr_completion:
        print(f"  FusionMethod:  {args.ablation_fusion}")
        print(f"  RecLossWeight: {args.rec_loss_weight}")
        print(f"  Temperature:   {args.temperature}")
        print(f"  CompletionCSWD:{not args.no_completion_cswd}")
        attr_cfg = getattr(args, 'attr_completion', None)
        if attr_cfg:
            print(f"  Targets:       {list(attr_cfg.keys())}")
    print(f"{'='*60}\n")

    main(args)