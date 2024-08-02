#!/bin/bash

#SBATCH --output=output_ehalmax.log
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load easybuild/4.8.2
module load cuda/12.3

## command line argument - run id
if [ $# -gt 0 ]; then
  runid=$1
  echo "Got run id ${runid} from the command line"
fi

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

if [ -z ${runid} ]; then
  echo "runid is unset"
  if ! uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh 5 ehal_max; then
    echo "Failed to run ehal_max."
    exit 1
  fi
else
  echo "runid is set to '${runid}'"
  if ! uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh 5 ehal_max "${runid}"; then
    echo "Failed to run ehal_max with runid '${runid}'."
    exit 1
  fi
fi


