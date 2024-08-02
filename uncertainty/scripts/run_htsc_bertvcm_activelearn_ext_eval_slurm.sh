#!/bin/bash

#SBATCH --output=output_eval.log
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

if [ $# -gt 1 ]; then
  batchsize=$2
  echo "Got batch size ${batchsize} from the command line"
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
  for ((e=1;e<=10;e++)); do
    echo "run eval ${e}"
    uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh all eval | \
      tee run_htsc_bertvcm_activelearn_ext_eval.log | \
      python uncertainty/src/save_result.py \
        htsc_bertvcm_activelearn_ext_5_r1e$e.json
  done
elif [ ! -z ${runid} ] && [ -z ${batchsize} ]; then
  echo "runid is set to '${runid}'"
  for ((e=1;e<=10;e++)); do
    echo "run eval ${e}"
    uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh all eval "${runid}" | \
      tee run_htsc_bertvcm_activelearn_ext_eval.log | \
      python uncertainty/src/save_result.py \
        htsc_bertvcm_activelearn_ext_5_${runid}e$e.json
  done
else 
  echo "runid is set to '${runid}' and batchsize is set to '${batchsize}'"
  for ((e=1;e<=10;e++)); do
    echo "run eval ${e}"
    uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh all eval "${runid}" "${batchsize}" | \
      tee run_htsc_bertvcm_activelearn_ext_eval.log | \
      python uncertainty/src/save_result.py \
        htsc_bertvcm_activelearn_ext_5_${runid}e$e.json
  done
fi




