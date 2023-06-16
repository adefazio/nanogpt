#!/bin/bash -i
echo $CONDA_DEFAULT_ENV
echo $CONDA_PREFIX
echo ==================================
echo $PATH
echo ==================================
echo $PYTHONPATH
conda activate pt2
module list
#echo "Environment variables"
#printenv
#source activate /public/apps/anaconda3/5.0.1/envs/fair_env_latest_py3
export JOB_DIR=$run_config_dir/job-$SLURM_ARRAY_TASK_ID
echo "Copying source to $JOB_DIR"
mkdir -p $JOB_DIR
rsync -a $run_config_dir/src/ $JOB_DIR/

cd $JOB_DIR
export PYTHONPATH=$PYTHONPATH:$JOB_DIR
pwd
exec python3 cluster/array_job.py $run_config_file_name
