#!/bin/bash

###
# Example to run this script
# 1. To run the initial training with 20% of the data (1/20% = 5)
#    uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh 5 init
# 2. To run the active learning with ehal (entropy-based heuristic active learning) - best case
#    uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh 5 ehal_max
# 3. To run the active learning with ehal (entropy-based heuristic active learning) - worst case
#    uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh 5 elah_max
# 4. To run the active learning with random
#    uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh 5 random
# 5. To evaluate the models
#    uncertainty/scripts/run_htsc_psvcm_activelearn_ext.sh all eval
###

###
# to run all pairs as follows:
###
# PYTHONPATH=uncertainty/src \
#   python uncertainty/src/htsc_ps_vcmdata_active_learn_train.py \
#     -c uncertainty/config/htscpsvcm/active/en_psvcm_1.0_im_1_gamma_0.5_train_0.9_sigma_0.ini -a init
# PYTHONPATH=uncertainty/src \
#   python uncertainty/src/htsc_ps_vcmdata_active_learn_train.py \
#     -c uncertainty/config/htscpsvcm/active/en_psvcm_1.0_im_1_gamma_0.5_train_0.9_sigma_0.ini -a ehal
# PYTHONPATH=uncertainty/src \
#   python uncertainty/src/htsc_ps_vcmdata_active_learn_train.py \
#     -c uncertainty/config/htscpsvcm/active/en_psvcm_1.0_im_1_gamma_0.5_train_0.9_sigma_0.ini -a elah
###

run_config_file="uncertainty/config/htscpsvcm/active/en_psvcm_1.0_im_1_gamma_0.5_train_0.9_sigma_0_ext.ini"

if [[ $# -eq 0 ]]; then 
	echo "Usage: $0 init_train_factor action"
	exit 0
fi

if [[ "$2" == "eval" ]]; then
	eval_action=$1
	action=$2
else
	init_train_factor=$1
	action=$2
fi

echo "init_train_factor=${init_train_factor}"
echo "action=${action}"

[ -f .env ] || (echo "unable to access from $(pwd)/.env" && exit 1)

. .env

function do_action() {
	if [[ $# -lt 2 ]]; then
		echo "do_action: missing parameter init_train_factor and/or action"
		exit 1
	fi
    init_train_factor=$1
	action=$2
	seed=$(( RANDOM ))
	echo -n "PYTHONPATH=$PYTHONPATH "
	echo -n "python uncertainty/src/htsc_ps_vcmdata_active_learn_train_ext.py "
	echo    "  -c  ${run_config_file} -i ${init_train_factor} -a ${action} -s ${seed}"
	PYTHONPATH=$PYTHONPATH python uncertainty/src/htsc_ps_vcmdata_active_learn_train_ext.py \
		-c  "${run_config_file}" -i "${init_train_factor}" -a "${action}" -s  "${seed}"
}

function do_eval() {
	echo -n "PYTHONPATH=$PYTHONPATH "
	echo -n "python uncertainty/src/htsc_ps_vcmdata_active_learn_eval_ext.py "
	echo    "  -c ${run_config_file} -a ${eval_action}"
	PYTHONPATH=$PYTHONPATH python uncertainty/src/htsc_ps_vcmdata_active_learn_eval_ext.py \
		-c  "${run_config_file}" -a "${eval_action}"
}

function do_action_with_check() {
	if [[ $# -lt 2 ]]; then
		echo "do_action: missing parameter init_train_factor and/or action"
		exit 1
	fi
    init_train_factor=$1
	action=$2
	case "$action" in
		init | ehal | elah | ehal_ratio | elah_ratio | ehal_max | elah_max | random )
			do_action "${init_train_factor}" "${action}"
			;;
		*)
			echo "Action $action not supported"
			exit 1
	esac
}

if [[ "${action}" != "eval" && "${action}" != "all" ]]; then
	do_action_with_check "${init_train_factor}" "$action"
fi

if [[ "${action}" == "all" ]]; then
	for action in init ehal elah; do
		echo "train for $action with init_train_factor=${init_train_factor}"
		do_action_with_check "${init_train_factor}" "$action"
		echo "completed training for $action with init_train_factor ${init_train_factor}"
	done
fi


if [[ "${action}" == "eval" ]]; then
	do_eval
fi
