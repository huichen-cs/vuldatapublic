#!/bin/bash

[ -f .env ] || (echo "unable to access from $(pwd)/.env" && exit 1)

. .env

n_runs=10
if [ $# -ge 1 ]; then
    n_runs=$1
    shift
else
    echo "Usage: $0 n_runs [eval]"
    exit 0
fi

train=1
if [ $# -ge 1 ]; then
    if [ "$1" == "eval" ]; then
        train=0
    elif [ "$1" == "train" ]; then
        train=1
    else
        echo "Usage: $0 n_runs [eval]"
        exit 0
    fi
fi

echo "n_runs = $n_runs train = $train"

portion=1.0
for ((i=0; i<n_runs; i++)); do
    echo "run no = $i"
    if [ $train -eq 1 ]; then
        echo -n "PYTHONPATH=$PYTHONPATH "
        echo -n "python uncertainty/src/htsc_ps_data_shift_train.py "
        echo    "  -c  uncertainty/config/uqde/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.ini"
        PYTHONPATH=$PYTHONPATH python uncertainty/src/htsc_ps_data_shift_train.py \
            -c  uncertainty/config/uqde/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.ini
    fi

    echo -n "PYTHONPATH=$PYTHONPATH "
    echo -n "python uncertainty/src/htsc_ps_data_shift_eval.py "
    echo -n "  -c  uncertainty/config/uqde/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.ini |"
    echo    "  python uncertainty/src/save_result.py en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.json"
    PYTHONPATH=$PYTHONPATH python uncertainty/src/htsc_ps_data_shift_eval.py \
        -c  uncertainty/config/uqde/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.ini | \
        python uncertainty/src/save_result.py en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.json
done
