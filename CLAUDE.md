# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation for "Convergence Analysis of Sequential Federated Learning on Heterogeneous Data". Compares Parallel Federated Learning (PFL/FedAvg) and Sequential Federated Learning (SFL/CWT) algorithms on non-IID data distributions.

## Commands

### Run FedAvg (Parallel FL)
```bash
python main_fedavg.py -m wvgg9k4 -d cifar10 -R 4000 -K 5 -M 500 -P 10 --partition exdir --alpha 1 10.0 --optim sgd --lr 0.1 --global-lr 1.0 --batch-size 20 --seed 1234 --clip 10 --eval-num 1000 --device 0 --save-model 0
```

### Run CWT (Sequential FL)
```bash
python main_cwt.py -m wvgg9k4 -d cifar10 -R 4000 -K 5 -M 500 -P 10 --partition exdir --alpha 1 10.0 --optim sgd --lr 0.03 --global-lr 1.0 --batch-size 20 --seed 1234 --clip 50 --eval-num 1000 --device 0 --save-model 0
```

### Run Quadratic Experiments
```bash
python quadratic.py -R 500 -K 2 -M 2 -P 2 --F1 0.5 0 --F2 0.5 0 --lr 0 --momentum 0 --weight-decay 0 --seed 0
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `-m` | Model: logistic, mlp, lenet5, cnnmnist, wvgg9k4, resnetii18 |
| `-d` | Dataset: mnist, fashionmnist, cifar10, cinic10 |
| `-R` | Total training rounds |
| `-K` | Local steps per client |
| `-M` | Total number of clients |
| `-P` | Clients participating per round |
| `--partition` | Data partition: iid, dir, exdir (Extended Dirichlet) |
| `--alpha` | Two parameters for Extended Dirichlet (C, concentration) |
| `--clip` | Gradient clipping max norm (use 10 for FedAvg, 50 for CWT) |
| `--eval-num` | Number of evaluation points (e.g., 1000 means evaluate every R/1000 rounds) |

## Architecture

```
sim/
├── algorithms/
│   ├── fedbase.py      # FedClient/FedServer for FedAvg
│   ├── cwtbase.py      # CWTClient/CWTServer for Sequential FL
│   └── splitbase.py    # Split Learning base classes
├── data/
│   ├── datasets.py     # Dataset builders with transforms
│   ├── partition.py    # IID, Dir, ExDir partitioning strategies
│   └── data_utils.py   # FedDataset wrapper
├── models/
│   ├── build_models.py # Model factory
│   └── *.py            # Model implementations (logistic, mlp, lenet5, cnn, vgg, resnet)
└── utils/
    ├── utils.py        # AverageMeter, accuracy, setup_seed, average_weights
    ├── optim_utils.py  # OptimKit, LrUpdater classes
    └── record_utils.py # Logging and experiment recording
```

## Model-Dataset Mapping

| Dataset | Models |
|---------|--------|
| MNIST | logistic, mlp, lenet5 |
| Fashion-MNIST | lenet5, cnnmnist |
| CIFAR-10/CINIC-10 | wvgg9k4, resnetii18 |

## Important Notes

- **No group normalization**: ResNet-18 without group norm (`resnetii18`) performs better than with group norm (`resnetgnii18`) in FL settings
- **Local dataset construction**: Use `Subset(overallset, indices)` (Way 1) rather than list comprehension to preserve data augmentation behavior
- **Gradient clipping**: FedAvg typically uses `--clip 10`, CWT uses `--clip 50`
- **Learning rates**: SFL/CWT generally requires lower learning rates than PFL/FedAvg (see grid search tables in README.md)
- **Pre-computed partitions**: Data partitions are stored in `partition/` as JSON files for reproducibility
- **Results output**: Experiment results saved to `save/` as CSV files with descriptive naming convention
