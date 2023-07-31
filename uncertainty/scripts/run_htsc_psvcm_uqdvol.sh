#!/bin/bash

[ -f .env ] || (echo "unable to access from $(pwd)/.env" && exit 1)

. .env

train=0
if [ $# -ge 1 ]; then
    if [ "$1" == "eval" ]; then
        train=0
    elif [ "$1" == "train" ]; then
        train=1
    else
        echo "Usage: $0 [train|eval]"
        exit 0
    fi
fi

if [ $train -eq 1 ]; then
    for portion in "1.0" "0.95" "0.9" "0.8" "0.6"; do
    # for portion in "0.6"; do
        echo -n "PYTHONPATH=$PYTHONPATH "
        echo -n "python uncertainty/src/htsc_ps_data_shift_train.py "
        echo    "  -c  uncertainty/config/uqdvol/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.ini"
        PYTHONPATH=$PYTHONPATH python uncertainty/src/htsc_ps_data_shift_train.py \
            -c  uncertainty/config/uqdvol/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.ini
    done
fi

for portion in "1.0" "0.95" "0.9" "0.8" "0.6"; do
    echo -n "PYTHONPATH=$PYTHONPATH "
    echo -n "python uncertainty/src/htsc_ps_data_shift_eval.py "
    echo -n "  -c  uncertainty/config/uqdvol/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.ini |"
    echo    "  python uncertainty/src/save_result.py en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.json"
    PYTHONPATH=$PYTHONPATH python uncertainty/src/htsc_ps_data_shift_eval.py \
        -c  uncertainty/config/uqdvol/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.ini | \
        python uncertainty/src/save_result.py en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_${portion}.json
done
