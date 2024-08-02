#!/bin/bash

#SBATCH --output=output_init.log
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load easybuild/4.8.2
module load cuda/12.3

[ -f .env ] || (echo "unable to access from $(pwd)/.env" && exit 1)

if [ $# -lt 3 ]; then
  echo "Usage: $0 model_idx active_learn_method data_selection_ratio"
  echo "Usage: $0 model_idx active_learn_method data_selection_ratio runid"
  echo "Usage: $0 model_idx active_learn_method data_selection_ratio runid batchsize"
  exit 1
fi

## command line argument - model index
if [ $# -gt 0 ]; then
  model_idx=$1
  echo "Got model_idx ${model_idx} from the command line"
fi

## command line argument - active learn method
if [ $# -gt 1 ]; then
  method=$2
  echo "Got model_idx ${method} from the command line"
fi


## command line argument - ratio
if [ $# -gt 2 ]; then
  ratio=$3
  echo "Got data select ratio ${ratio} from the command line"
fi

## command line argument - run id
if [ $# -gt 3 ]; then
  runid=$4
  echo "Got run id ${runid} from the command line"
fi

## command line argument - batch size
if [ $# -gt 4 ]; then
  batchsize=$5
  echo "Got batch size ${batchsize} from the command line"
fi

linevul_input_file="data/linevul/splits/commit_train.csv"
if [ $# -gt 5 ]; then
  linevul_input_file=$6
  echo "Got lineveul input file ${linevul_input_file} from the command line"
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

if [ -z "${runid}" ]; then
  echo "runid is unset"
  if ! uncertainty/scripts/run_htsc_bertvcm_activelearn_ext_linevul_by_commit.sh \
    "${model_idx}" "${method}" "${ratio}"; then
    echo "Failed to run run_htsc_bertvcm_activelearn_ext_linevul.sh"
    echo "model_idx: ${model_idx}, method: ${method}, ratio: ${ratio}"
    exit 1
  fi
elif [ -n "${runid}" ] && [ -z "${batchsize}" ]; then
  echo "runid is set to '${runid}'"
  if ! uncertainty/scripts/run_htsc_bertvcm_activelearn_ext_linevul_by_commit.sh \
    "${model_idx}" "${method}" "${ratio}" "${runid}"; then
    echo "Failed to run run_htsc_bertvcm_activelearn_ext_linevul.sh"
    echo "model_idx: ${model_idx}, method: ${method}, ratio: ${ratio}, runid: ${runid}"
    exit 1
  fi
elif [ -n "${runid}" ] && [ -n "${batchsize}" ] && [ -z "${linevul_input_file}" ]; then
  echo "runid is set to '${runid}'"
  echo "runid is set to '${runid}', batchsize is set to '${batchsize}'"
  if ! uncertainty/scripts/run_htsc_bertvcm_activelearn_ext_linevul_by_commit.sh \
    "${model_idx}" "${method}" "${ratio}" "${runid}" "${batchsize}"; then
    echo "Failed to run run_htsc_bertvcm_activelearn_ext_linevul.sh"
    echo "model_idx: ${model_idx}, method: ${method}, ratio: ${ratio}, runid: ${runid}, batchsize: ${batchsize}"
    exit 1
  fi
else
  echo "runid is set to '${runid}'"
  echo -n "runid is set to '${runid}', batchsize is set to '${batchsize}' "
  echo    "linevul_input_file is set to '${linevul_input_file}'"
  if ! uncertainty/scripts/run_htsc_bertvcm_activelearn_ext_linevul_by_commit.sh \
    "${model_idx}" "${method}" "${ratio}" "${runid}" "${batchsize}" "${linevul_input_file}"; then
    echo "Failed to run run_htsc_bertvcm_activelearn_ext_linevul.sh"
    echo -n "model_idx: ${model_idx}, method: ${method}, ratio: ${ratio}, runid: ${runid}, "
    echo    "batchsize: ${batchsize}, linevul_input_file: ${linevul_input_file}"
    exit 1
  fi
fi
echo "Done."

