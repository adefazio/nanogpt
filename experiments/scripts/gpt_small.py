import os
import sys
import spawn_dist
import spawn_single

runinfo = {
    'config': 'gpt2_small_config.py',
    # Should be set by config script correctly.
    # 'batch_size': 8,
    # 'gradient_accumulation_steps': 6,

    # 'optimizer_name': 'sophiag',
    # 'learning_rate': 3e-4,
    # 'weight_decay': 2e-1,

    'optimizer_name': 'adamw',
    'learning_rate': 6e-4,
    'weight_decay': 2e-1, # They seem to use 0.1 for Adam?
}

if __name__ == "__main__":
    spawn_single.run(runinfo)