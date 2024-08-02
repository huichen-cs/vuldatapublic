#!/bin/bash

#SBATCH --output=output.log
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load easybuild/4.8.2
module load cuda/12.3

workdirbasename=$(basename "$(pwd)")
if [ "$workdirbasename" != "vuldata" ]; then
  echo "Please run this script from the vuldata directory."
  exit 1
fi

if [ ! -d "vuldatavenv" ]; then
    if ! python -m venv "vuldatavenv"; then
      echo "Failed to create virtual environment."
      exit 1
    fi
    if ! pip install -r uncertainty/requirements.txt; then
      echo "Failed to install requirements."
      exit 1
    fi 
fi

if ! source vuldatavenv/bin/activate; then
  echo "Failed to activate virtual environment."
  exit 1
else
  echo "activated virtual environment vuldatavenv"
fi

if ! uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh 5 init; then
  echo "Failed to run init."
  exit 1
fi

if ! uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh 5 ehal_max; then
 echo "Failed to run ehal_max."
 exit 1
fi

if ! uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh 5 elah_max; then
    echo "Failed to run elah_max."
    exit 1
fi

if ! uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh 5 random; then
    echo "Failed to run random."
    exit 1
fi  

for ((e=1;e<=10;e++)); do
  uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh all eval | \
    tee run_htsc_psvcm_activelearn_ext_eval.log | \
    python uncertainty/src/save_result.py \
      htsc_psvcm_activelearn_ext_5_r1e$e.json
done
