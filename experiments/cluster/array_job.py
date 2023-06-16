import os
from subprocess import call
import sys
import glob
import pickle
import importlib
import signal
import time

def termHandler(signum, frame):
    """
        Slurm preemption sends a SIGTERM before the SIGUSR1, to give you a warning
        that the process is going to be preempted. This needs to be caught, otherwise
        the process will exit early.
    """
    print("SIGTERM caught and ignored", flush=True)

def requeueHandler(signum, frame):
    """
        A USR1 signal is sent by slurm if the timelimit of the job is reached
        or if the job is about to be preempted
    """
    print('Signal received', signum, time.time(), flush=True)

    job_id = os.environ['SLURM_JOB_ID']
    print(f'requeuing job {job_id}', flush=True)
    os.system(f'scontrol requeue {job_id}')

    sys.exit(0)

def array_job(run_configs):
    cwd = os.getcwd()
    slurm_arrayid = int(os.getenv('SLURM_ARRAY_TASK_ID', -1))
    total_jobs = int(os.getenv('SLURM_ARRAY_TASK_COUNT', -1))
    job_id = int(os.getenv('SLURM_JOBID', -1))
    job_name = os.getenv('SLURM_JOB_NAME', -1)
    job_nodes = os.getenv('SLURM_NODELIST', -1)
    node_name = os.getenv('SLURMD_NODENAME', -1)
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
    python_path = os.getenv('PYTHONPATH', '')

    print(f"Current working directory: {cwd}")
    print(f"jobid: {job_id} ({job_name})")
    print(f"SLURM ARRAY ID: {slurm_arrayid}")
    print(f"SLURM total jobs: {total_jobs}")
    print(f"Node {node_name} of [{job_nodes}]")
    print(f"cuda_visible_devices: {cuda_visible_devices}")

    print(f"sys.path: {sys.path}")
    print(f"PYTHONPATH: {python_path}")

    if slurm_arrayid == -1:
        raise Exception("array id not set for job")

    if slurm_arrayid >= len(run_configs):
        raise Exception(f"arrayid is {slurm_arrayid} when only {njobs} configurations are given")

    print("Launching for arrayid: {}".format(slurm_arrayid))
    runargs = run_configs[slurm_arrayid]

    signal.signal(signal.SIGUSR1, requeueHandler)
    signal.signal(signal.SIGTERM, termHandler)

    #####################
    path = "python3"
    args = [
        "python3",
        "train.py",
    ]

    ignores = [
        'cluster_run_name',
        'partition',
        'stdout_log_file',
        'gpus',
        'time_string',
    ]

    #cofig=config/train_gpt2_small_adam.py

    for field, value in runargs.items():
        if field == "config":
            args.append(value)
        elif field in ignores:
            pass
        else:
            args.append(f"--{field}")
            if value is not None:
                args.append(str(value))

    print("Running:")
    print(" ".join(args))
    print("")
    sys.stdout.flush()
    os.chdir("fairseq")
    print(f"Current working dir: {os.getcwd()}")
    sys.stdout.flush()
    os.spawnvp(os.P_WAIT, path, args)

if __name__ == "__main__":
    print(f"array_job pid: {os.getpid()}")
    run_config_file = sys.argv[1]
    print(f"Config file: {run_config_file}")
    run_configs = pickle.load(open(run_config_file, 'rb'))
    array_job(run_configs)
