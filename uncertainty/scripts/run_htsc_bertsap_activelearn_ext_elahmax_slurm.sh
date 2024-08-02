#!/bin/bash

#SBATCH --output=output_elahmax.log
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

if ! which nvdia-smi > /dev/null 2>&1; then
  pyvenv="vuldatavenv"
else
  if ! (nvidia-smi -L | grep -q H100); then
    pyvenv="vuldatavenvtorch2"
  else
    pyvenv="vuldatavenv"
  fi
fi

if [ ! -d "${pyvenv}" ]; then
    if ! python -m venv "${pyvenv}"; then
      echo "Failed to create virtual environment ${pyvenv}."
      exit 1
    fi
    if ! source "${pyvenv}/bin/activate"; then
      echo "Failed to activate virtual environment ${pyvenv}."
      exit 1
    else
      echo "activated virtual environment vuldatavenv ${pyvenv}"
    fi
    if ! pip install -r "uncertainty/requirements_${pyvenv}.txt"; then
      echo "Failed to install requirements uncertainty/requirements_${pyvenv}.txt."
      exit 1
    fi
else
  if ! source "${pyvenv}/bin/activate"; then
    echo "Failed to activate virtual environment ${pyvenv}."
    exit 1
  else
    echo "activated virtual environment vuldatavenv ${pyvenv}"
  fi
fi

if [ -z ${runid} ]; then
  echo "runid is unset"
  if ! uncertainty/scripts/run_htsc_bertsap_activelearn_ext.sh 5 elah_max; then
    echo "Failed to run elah_max."
    exit 1
  fi
else
  echo "runid is set to '${runid}'"
  if ! uncertainty/scripts/run_htsc_bertsap_activelearn_ext.sh 5 elah_max "${runid}"; then
    echo "Failed to run elah_max with runid '${runid}'."
    exit 1
  fi
fi

