#!/bin/bash

help() {
	echo "Usage: run_hmsc_bertvcm_gn_shift_eval [-f|--n_first <n_first>]"
	echo "          [-l|--n_last <n_last>] [-s|--step_size <step_size>]"
	echo "          [-b|--im_ratio <im_ratio>] [-m|--uq_method <uq_method>]"
	echo "          [-c|--select_criteria <selection_criteria>]"
	echo "       the configuration file and the checkpoint for all runs"
	echo "       must exist to evaluate UQ metrics."
	echo "       selction_criteria is only relevant when the uq_method is"
	echo "       either vanilla or dropout."
}

if [ $# -eq 0 ]; then
	help
	exit 0
fi

n_first=0
n_last=20
step_size=0.01
im_ratio=1
uq_method="undefined"
if ! VALID_ARGS=$(getopt -o f:l:s:b:m:h:c --long n_first:,n_last:,step_size:,im_ratio:,uq_method:,select_criteria:,help -- "$@"); then
    exit 1;
fi

eval set -- "${VALID_ARGS}" # gives us $1, $2, $3 ...
while true; do
	case "$1" in
		-f | --n_first)
			n_first=$2
			shift 2
			;;
		-l | --n_last)
			n_last=$2
			shift 2
			;;
		-s | --step_size)
			step_size=$2
			shift 2;
			;;
		-b | --im_ratio)
			im_ratio=$2
			shift 2
			;;
		-m | --uq_method)
			uq_method=$2
			shift 2
			;;
		-c | --select_criteria)
			select_criteria=$2
			shift 2
			;;
		-h | --help)
			help
			exit 0
			;;
		--) shift;
			break
			;;
	esac
done

config_root=uncertainty/config/bertps/shift
ckpt_root=uq_testdata_ckpt/bertps/shift/

eval_from_ckpt() 
{
	if [ $# -lt 4 ]; then
		echo "ERROR: eval_from_ckpt(): insufficient function parameters, expected n_first n_last im_ratio uq_type"
		echo "ERROR: example: eval_from_ckpt 0 20 1 vanilla"
		return 1  #FAILURE
	fi

	_n_first=$1
	_n_last=$2
	_im_ratio=$3
	_uq_type=$4

	if [ $# -gt 4 ]; then
		_select_criteria=$5
	fi

	init_ini=bertps_1.0_im_${im_ratio}_train_0.9_sigma_0.ini
	for ((n=_n_first; n<=_n_last; n++)); do
		# _sigma=$(echo "${init_ini}" | cut -d'_' -f11 | sed -e 's/.ini//g' | awk -v n=$n '{s=$0; if (n > 0) s=n/100+$0; print s;}')
		if [ "${step_size}" == "0.01" ]; then
			_sigma=$(echo "${init_ini}" | cut -d'_' -f8 | sed -e 's/.ini//g' | awk -v n="$n" '{s=$0; if (n > 0) s=n/100+$0; print s;}')
		else
			_sigma=$(echo "${init_ini}" | cut -d'_' -f8 | sed -e 's/.ini//g' | awk -v n="$n" -v t="${step_size}" '{s=$0; if (n > 0) s=n*t+$0; print s;}')
		fi
		echo "INFO: sigma: ${_sigma}"
		ckpt_d=${ckpt_root}/en_bertps_1.0_im_${im_ratio}_train_0.9_sigma_${_sigma}
		ini_fp=${config_root}/bertps_1.0_im_${im_ratio}_train_0.9_sigma_${_sigma}.ini

		if [ ! -f "${ini_fp}" ]; then 
			echo "ERROR: experiment configuration ${ini_fp} inaccessible"
			exit 1
		fi

		if [ ! -d "${ckpt_d}" ]; then
			echo "ERROR: experiment checkpoint at ${ckpt_d} inaccessbile"
			exit 1
		fi

		case "${_uq_type}" in
			"vanilla")
				_py_key="vanilla"
				# _select="--select ${_select_criteria}"
				_select="${_select_criteria}"
				;;
			"dropout")
				_py_key="dropout"
				# _select="--select ${_select_criteria}"
				_select="${_select_criteria}"
				;;
			"ensemble")
				_py_key="ensemble"
				_select=""
				;;
			*)
				echo "ERROR: unsupported uq_type ${_uq_type}"
				echo "ERROR: supported uq_type's are vanilla, dropout, and ensemble"
				return 1
				;;
		esac
 		if [ -z "${_select}" ]; then
		echo "INFO: PYTHONPATH=uncertainty/src/ python uncertainty/src/hmsc_bert_vcmdata_shift_${_py_key}_eval.py -c ${ini_fp}"
 			if ! PYTHONPATH=uncertainty/src/ python "uncertainty/src/hmsc_bert_vcmdata_shift_${_py_key}_eval.py" -c "${ini_fp}"; then
 				echo "ERROR: uncertainty/src/hmsc_bert_vcmdata_shift_${_py_key}_eval.py failed"
 				return 1
 			fi	
 		else
		echo "INFO: PYTHONPATH=uncertainty/src/ python uncertainty/src/hmsc_bert_vcmdata_shift_${_py_key}_eval.py -c ${ini_fp} --select ${_select}"
 			if ! PYTHONPATH=uncertainty/src/ python "uncertainty/src/hmsc_bert_vcmdata_shift_${_py_key}_eval.py" -c "${ini_fp}" --select "${_select}"; then
 				echo "ERROR: uncertainty/src/hmsc_bert_vcmdata_shift_${_py_key}_eval.py failed"
 				return 1
 			fi	
 		fi	
	done
	return 0
}

if ! eval_from_ckpt "${n_first}" "${n_last}" "${im_ratio}" "${uq_method}" "${select_criteria}"; then
	echo "ERROR: eval_from_ckpt \"${n_first}\" \"${n_last}\" \"${im_ratio}\" \"${uq_method}\" failed"
	exit 1
fi

exit 0
