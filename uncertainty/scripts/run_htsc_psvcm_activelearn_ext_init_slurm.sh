#!/bin/bash

#SBATCH --output=output_random.log
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

