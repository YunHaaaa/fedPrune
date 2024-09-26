# Federated Dynamic Sparse Training (FedDST)

This experiment code accompanies the paper

> Sameer Bibikar, Haris Vikalo, Zhangyang Wang, and Xiaohan Chen, "Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better," 2021.

## Dependencies
- Python 3.6 or greater
- PyTorch, torchvision
- tqdm

Run `git submodule init` followed by `git submodule update` to download the dataset code we use.

## Examples

| Experiment | Command line |
| ---------- | ------------ |
| FedDST on MNIST (S=0.8) | `python3 dst.py --dataset mnist --sparsity 0.8 --readjustment-ratio 0.05` |
| FedAvg on CIFAR-10 | `python3 dst.py --dataset cifar10 --sparsity 0.0` |
| FedProx on CIFAR-10 (mu = 1) | `python3 dst.py --dataset cifar10 --sparsity 0.0 --prox 1` |
| FedDST on CIFAR-10 (S=0.8, alpha=0.01, R_adj=15) | `python3 dst.py --dataset cifar10 --sparsity 0.8 --readjustment-ratio 0.01 --rounds-between-readjustments 15` |
| FedDST on CIFAR-100 (S=0.5, alpha=0.01, R_adj=10) | `python3 dst.py --dataset cifar100 --sparsity 0.5 --readjustment-ratio 0.01 --distribution dirichlet --beta 0.1` |
| FedDST+FedProx on CIFAR-10 (S=0.8, alpha=0.01, R_adj=15, mu=1) | `python3 dst.py --dataset cifar10 --sparsity 0.8 --readjustment-ratio 0.01 --rounds-between-readjustments 15 --prox 1` |
| RandomMask on MNIST (S=0.8) | `python3 dst.py --dataset mnist --sparsity 0.8 --readjustment-ratio 0.0` |
| PruneFL on MNIST | `python3 prunefl.py --dataset mnist --rounds-between-readjustments 50 --initial-rounds 1000` |
| FedDST on MNIST (S=0.8) | `python3 dst.py --dataset mnist --sparsity 0.8 --readjustment-ratio 0.05` |
| Ours on MNIST (S=0.8) | `python3 ours.py --dataset mnist --sparsity 0.8 --readjustment-ratio 0.05 -o ours.log` |
| CS on CIFAR-10 (S=0.8) | `python3 cs.py --dataset cifar10 --sparsity 0.8 --readjustment-ratio 0.05 -o cs_cifar_0.8.log` |
| Dhr on CIFAR-10 (S=0.8) | `python3 dst_hard_retrain.py --dataset cifar10 --sparsity 0.8 --pruning-type soft --outfile dhr_cifar_0.8_soft.log` |
| De on CIFAR-10 (S=0.8) | `python3 dst_ensemble.py --dataset cifar10 --sparsity 0.8  --outfile de_cifar_0.8.log` |
| Dmr on CIFAR-10 (S=0.8) | `python3 dst_~mask_retrain.py --dataset cifar10 --sparsity 0.8 --pruning-type soft --type-value 5 --outfile dmr_cifar_0.8_soft_7:3_tv5.log` |