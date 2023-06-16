import os
import sys
from pathlib import Path
from experiments.cluster.run_on_cluster import run_on_cluster

base_config = {
    'init_from': 'resume',
    'config': 'gpt2_small_config.py',

    'optimizer_name': 'sophiag',
    'learning_rate': 3e-4,
    'weight_decay': 2e-1,
}

run_configs = [base_config]

if __name__ == "__main__":
    run_on_cluster(
        run_configs,
        gpus=8,
        partition="learnlab,learnfair",
        time_string="72:00:00")
