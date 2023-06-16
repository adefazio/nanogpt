import os
import sys
from pathlib import Path
from experiments.cluster.run_on_cluster import run_on_cluster

base_config = {
    'init_from': 'resume',
    'config': 'gpt2_small_config.py',

    'optimizer_name': 'adamw',
    'learning_rate': 6e-4,
    'weight_decay': 1e-1,
    'beta1': 0.9,
    'beta2': 0.95,
    'min_lr': 3e-5,
}

run_configs = [base_config]

if __name__ == "__main__":
    run_on_cluster(
        run_configs,
        gpus=8,
        partition="learnlab,learnfair",
        time_string="72:00:00")
