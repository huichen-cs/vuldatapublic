#!/bin/bash

if [[ $# -eq 0 ]]; then 
	echo "Usage: $0 action"
	exit 0
fi
action=$1

[ -f .env ] || (echo "unable to access from $(pwd)/.env" && exit 1)

. .env

function do_action() {
	if [[ $# -lt 1 ]]; then
		echo "do_action: missing parameter action"
		exit 1
	fi
	action=$1
	echo -n "PYTHONPATH=$PYTHONPATH "
	echo -n "python uncertainty/src/ps_data_active_learn_test.py "
	echo    "  -c  uncertainty/config/active/v3/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_1.0.ini -a ${action}"
	PYTHONPATH=$PYTHONPATH python uncertainty/src/ps_data_active_learn_test.py \
		-c  uncertainty/config/active/v3/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_1.0.ini -a "${action}"
}

function do_eval() {
	echo -n "PYTHONPATH=$PYTHONPATH "
	echo -n "python uncertainty/src/htsc_ps_vcmdata_active_learn_eval.py "
	echo    "  -c  uncertainty/config/active/v3/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_1.0.ini"
	PYTHONPATH=$PYTHONPATH python uncertainty/src/htsc_ps_vcmdata_active_learn_eval.py \
		-c  uncertainty/config/active/v3/en_vcm_1.0_im_1_train_0.9_sigma_0.0_sft_1.0.ini
}

function do_action_with_check() {
	if [[ $# -lt 1 ]]; then
		echo "do_action: missing parameter action"
		exit 1
	fi
	action=$1
	case "$action" in
		init | ehal | elah | ehah | elal | aleh | ahel | aheh | alel)
			do_action "${action}"
			;;
		*)
			echo "Action $action not supported"
			exit 1
	esac
}

if [[ "${action}" != "eval" && "${action}" != "all" ]]; then
	do_action_with_check "$action"
fi

if [[ "${action}" == "all" ]]; then
	for action in init ehal elah ehah elal aleh ahel aheh alel; do
		echo "train for $action"
		do_action_with_check "$action"
		echo "completed training for $action"
	done
fi


if [[ "${action}" == "eval" ]]; then
	do_eval
fi
