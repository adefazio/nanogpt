import os
from subprocess import call
import sys
import glob
import tempfile
import pickle
from datetime import datetime

def run_on_cluster(run_configs,
        maximum_at_once=-1, gpus=1, partition="learnlab,learnfair",
        time_string="48:00:00",
        require_32gb_gpus=False,
        comment="nanogpt",
        checkpoint_dir="/checkpoint/adefazio/nanogpt"):
    njobs = len(run_configs)
    if maximum_at_once == -1 or maximum_at_once > njobs:
        maximum_at_once = njobs
    if len(run_configs) == 0:
        raise Exception("No jobs to run in passed list")

    cluster_run_name = run_configs[0].get("cluster_run_name", "crun")

    datestamp = f"{datetime.now():%Y%m%d_%H%M%S}"

    slurm_arrayid = int(os.getenv('SLURM_ARRAY_TASK_ID', -1))

    # In master process, kick off the job
    if slurm_arrayid != -1:
        raise Exception("Cluster runner doesn't support nesting")
    print(f"Queueing {njobs} jobs")
    base_path = f"{checkpoint_dir}"


    # Directory of logs for just the latest/currently running cluster job.
    # Useful for monitoring
    os.makedirs(base_path + "/latest-logs", exist_ok=True)
    stdout_log_file_prefix = f"{base_path}/latest-logs/log-{cluster_run_name}"
    stdout_log_file = stdout_log_file_prefix + "-%a.log"

    for filename in glob.glob(stdout_log_file_prefix + "*"):
        os.remove(filename)

    ## This information is needed by the jobs for resuming from a checkpoint
    for i in range(len(run_configs)):
        run_configs[i]["gpus"] = gpus
        run_configs[i]["time_string"] = time_string
        run_configs[i]["partition"] = partition
        run_configs[i]["stdout_log_file"] = stdout_log_file

    checkpoint_path = f"{base_path}/{cluster_run_name}"
    os.makedirs(checkpoint_path, exist_ok=True)

    run_config_dir = tempfile.mkdtemp(dir=checkpoint_path, prefix=datestamp)
    run_config_file_name = f"{run_config_dir}/runconfig.pkl"
    with open(run_config_file_name, 'wb') as run_config_file:
        pickle.dump(run_configs, run_config_file, pickle.HIGHEST_PROTOCOL)
    run_config_file.close()
    print(f"Run configuration saved to {run_config_file_name}")

    print(f"Making copy of source tree to {run_config_dir}/src")
    os.system(f"rsync -am --include='*/' --include='*.py' --include='*.sh' --include='*.so' --include='*.pyd' --exclude='*' . {run_config_dir}/src")

    if require_32gb_gpus:
        constraint = "#SBATCH --constraint=volta32gb\n"
    else:
        constraint = ""

    sbatch_script = f"""#!/bin/bash
#SBATCH --array=0-{njobs-1}%{maximum_at_once}
#SBATCH --time={time_string}
#SBATCH --comment='NanoGPT optimization experiments'
#SBATCH --cpus-per-gpu 5
#SBATCH --gpus-per-task {gpus}
#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --mem-per-gpu 50G
#SBATCH --job-name={cluster_run_name}
#SBATCH --comment='{comment}'
#SBATCH --chdir={run_config_dir}
#SBATCH --partition={partition}
#SBATCH --output={stdout_log_file}
#SBATCH --error={stdout_log_file}
#SBATCH --signal=USR1@60
#SBATCH --export=PATH={os.getenv('PATH')}:/public/slurm/17.02.6/bin
{constraint}
echo "SBATCH"
srun --label --export=ALL src/cluster/cluster_bootstrap.sh
"""

    script_path = f"{run_config_dir}/sbatch_script.sh"
    with open(script_path, "w") as script_file:
        script_file.write(sbatch_script)
    export=f"--export=run_config_dir={run_config_dir},run_config_file_name={run_config_file_name}"
    cmd = ["sbatch", "-N 1", export, script_path]

    print(" ".join(cmd))
    os.system(" ".join(cmd))

    print("")
    print(f"To view latest logs:   tail -F {stdout_log_file_prefix}-0.log")