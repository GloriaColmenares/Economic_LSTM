#!/bin/bash
#SBATCH --job-name=paper_model
#SBATCH --nodes=1
#SBATCH --partition=geospark-long
#SBATCH --ntasks-per-node=32
#SBATCH --mem=512G
#SBATCH --error=job-%j-error.out
#SBATCH --output=job-%j-out.out
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/home/glo54486/Economic_LSTM_XGboost/01_Python_code


#HOW to submit:

#WITHOUT FEATURES
#sbatch --job-name=rw_sarma submit-SLURM.sh statistical_methods.rw_sarma
#sbatch --job-name=xgboost submit-SLURM.sh machine_learning_methods.xgboost

#WITH FEATURES
#FEATURE_SET=1_4 sbatch --job-name=rwx_14ar submit-SLURM.sh statistical_methods.rw_armax
#FEATURE_SET=4 sbatch --job-name=xgb4 submit-SLURM.sh machine_learning_methods.xgboost

source /ceph/home/glo54486/.venv/bin/activate

echo "============================================================"
echo "SLURM JOB METADATA"
echo "============================================================"
echo "Job started at          : $(date)"
echo "Job ID                  : ${SLURM_JOB_ID:-not_set}"
echo "Job name                : ${SLURM_JOB_NAME:-not_set}"
echo "Partition               : ${SLURM_JOB_PARTITION:-not_set}"
echo "Submit host             : ${SLURM_SUBMIT_HOST:-not_set}"
echo "Submit directory        : ${SLURM_SUBMIT_DIR:-not_set}"
echo "Working directory       : $(pwd)"
echo "Node list               : ${SLURM_JOB_NODELIST:-not_set}"
echo "Current host            : $(hostname)"
echo "Nodes allocated         : ${SLURM_JOB_NUM_NODES:-not_set}"
echo "Tasks per node          : ${SLURM_TASKS_PER_NODE:-not_set}"
echo "CPUs per task           : ${SLURM_CPUS_PER_TASK:-not_set}"
echo "NTASKS                  : ${SLURM_NTASKS:-not_set}"
echo "Memory requested        : ${SLURM_MEM_PER_NODE:-not_set}"
echo "Array job ID            : ${SLURM_ARRAY_JOB_ID:-not_set}"
echo "Array task ID           : ${SLURM_ARRAY_TASK_ID:-not_set}"
echo "Module to run           : $1"
echo "FEATURE_SET             : ${FEATURE_SET:-not_set}"
echo "Python executable       : $(which python3)"
echo "Python version          : $(python3 --version 2>&1)"
echo "============================================================"

# Optional git logging for exact code version
if command -v git >/dev/null 2>&1; then
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "GIT REPOSITORY INFO"
        echo "Git commit              : $(git rev-parse HEAD)"
        echo "Git branch              : $(git rev-parse --abbrev-ref HEAD)"
        echo "Git status (short)      :"
        git status --short
        echo "============================================================"
    fi
fi

# Start timer
START_TIME=$(date +%s)

# Run the model
python3 -m "$1"

# Capture exit code
EXIT_CODE=$?

# End timer
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "============================================================"
echo "JOB FINISHED"
echo "============================================================"
echo "Job finished at         : $(date)"
echo "Exit code               : ${EXIT_CODE}"
echo "Total runtime (seconds) : ${ELAPSED}"
printf "Total runtime (hh:mm:ss): %02d:%02d:%02d\n" "${HOURS}" "${MINUTES}" "${SECONDS}"
echo "============================================================"

exit ${EXIT_CODE}



