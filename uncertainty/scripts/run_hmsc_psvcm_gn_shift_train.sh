#!/bin/bash

function help() {
	echo "Usage: run_gn_shift [-f|--n_first <n_first>] [-l|--n_last <n_last>]"
	echo "           [-s|--step_size <step_size>] [-b|--im_ratio <im_ratio>]"
	echo "           [-r|--rerun] [-d|--new_data] [-p|--reproduce <true|false>]"
	echo "       the configuration file and the checkpoint for n_first must exists."
	echo "       they will be copied to next configuration and checkpoint, however,"
	echo "       the first configuration and checkpoint will not be evaluated or "
	echo "       trained unless -r or --rerun is given."
	echo "       -f: initial sigma (shift intensity) multiplier, if 0, initial sigma is 0 * 0"
	echo "       -l: final sigma (shift intensity) multiplier, if 20, last sigma is 20 * 0.01"
	echo "       -s: step size for sigma, default 0.01 or unless given"
	echo "       -b: imbalance ratio, if 3, vulnerability:non-vulnerability patches = 1:3"
	echo "       -r: whether to retrain model for checkpoint with sigma = 0"
	echo "       -d: whether to regenerate data in the intial checkpoint, which implies reurn (-r)"
	echo "       -p: true or false or randomness control (repeatability)"
}

if [ $# -eq 0 ]; then
	help
	exit 0
fi

n_first=0
n_last=20
step_size=0.01
im_ratio=1
rerun=false
new_data=false
reproduce=true
if ! VALID_ARGS=$(getopt -o f:l:b:s:rdp:h --long n_first:,n_last:,step_size:,im_ratio:,rerun,new_data,reproduce:,help -- "$@"); then
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
			shift 2
			;;
		-b | --im_ratio)
			im_ratio=$2
			shift 2
			;;
		-r | --rerun)
			rerun=true
			shift 1
			;;
		-d | --new_data)
			new_data=true
			shift 1
			;;
		-p | --reproduce)
			if [ "$2" == "true" ]; then
				reproduce=true
			elif [ "$2" == "false" ]; then
				reproduce=false
			else
				help
				exit 0
			fi
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

function cleanup()
{
    if [ -z ${ckpt_d+x} ]; then
		echo "INFO: ckpt_d is unset, nothing to clean up"
	else
		if [ "${ckpt_d}" != "${init_ckpt_d}" ]; then
			rm -rvf "${ckpt_d}"
			echo "INFO: removed checkpoint '${ckpt_d}'"
		fi
	fi
}

function clean_data()
{
	if [ $# -ne 1 ]; then
		echo "ERROR: clean_data(): expected ckpt_d, but not provided"
		return 1
	fi

	_ckpt_d=$1
	if [ -z ${_ckpt_d+x} ]; then
		return 0
	fi

	_file_list=(dataset_ps_columns.pickle  dataset_test.pickle  dataset_train.pickle  dataset_val.pickle)
	for _name in "${_file_list[@]}"; do
		_fpath="${_ckpt_d}/${_name}"
		if [ -f "${_fpath}" ]; then
			if ! rm "${_fpath}"; then
				echo "ERROR: clean_data(): rm \"${_fpath}\" failed"
				return 1
			fi
		fi
	done
	return 0
}

function set_reproduce()
{
	if [ $# -ne 2 ]; then
		echo "ERROR: set_reproduce() expected two arguments"
		return 1
	fi

	_ini_fp=$1
	_reproduce=$2

	if [ "${_reproduce}" == false ]; then
		if grep -q "\[reproduce\]" "${_ini_fp}"; then
			sed --in-place=.bu -e 's/\[reproduce\]/\[ignore-reproduce\]/g' "${_ini_fp}"
		fi
	else
		if grep -q "\[ignore-reproduce\]" "${_ini_fp}"; then
			sed --in-place=.bu -e 's/\[ignore-reproduce\]/\[reproduce\]/g' "${_ini_fp}"
		fi

		if ! grep -q "\[reproduce\]" "${_ini_fp}"; then
			echo "ERROR: ini file \"${_ini_fp}\" misses \"[reproduce]\" section"
			return 1
		fi
	fi
	return 0
}

function make_init_ini_file()
{
	if [ $# -ne 3 ]; then
		echo "ERROR: make_init_ini_file() expected three arguments"
		return 1
	fi

	_init_ini_im_1_fp=$1
	_init_ini_fp=$2
	_im_ratio=$3

	if [ -f "${_init_ini_fp}" ]; then
		echo "ERROR: ini file \"${_init_ini_fp}\" already exists, not to overwrite."
		return 1
	fi

	sed -e "s/imbalance_ratio = 1/imbalance_ratio = ${_im_ratio}/g" "${_init_ini_im_1_fp}" > "${_init_ini_fp}"
	if ! grep -q "imbalance_ratio = ${_im_ratio}" "${_init_ini_fp}"; then
		echo "ERROR: unexpected imbalance_ratio in \"${_init_ini_fp}\""
		rm "${_init_ini_fp}"
		return 1
	fi
	return 0
}


trap cleanup SIGINT

config_root=uncertainty/config/shift
ckpt_root=uq_testdata_ckpt/shift/rpd/
init_ini_im_1=en_vcm_1.0_im_1_gamma_0.5_train_0.9_sigma_0.ini
init_ini=en_vcm_1.0_im_${im_ratio}_gamma_0.5_train_0.9_sigma_0.ini
for ((n=n_first; n<=n_last; n++)); do
	if [ "${step_size}" == "0.01" ]; then
		sigma=$(echo "${init_ini}" | cut -d'_' -f11 | sed -e 's/.ini//g' | awk -v n="$n" '{s=$0; if (n > 0) s=n/100+$0; print s;}')
	else
		sigma=$(echo "${init_ini}" | cut -d'_' -f11 | sed -e 's/.ini//g' | awk -v n="$n" -v t="${step_size}" '{s=$0; if (n > 0) s=n*t+$0; print s;}')
	fi
	echo "INFO: sigma: ${sigma}"
	ckpt_d=${ckpt_root}/en_vcm_1.0_im_${im_ratio}_gamma_0.5_train_0.9_sigma_${sigma}
	ini_fp=${config_root}/en_vcm_1.0_im_${im_ratio}_gamma_0.5_train_0.9_sigma_${sigma}.ini
	init_ini_fp=${config_root}/${init_ini}
	init_ini_im_1_fp=${config_root}/${init_ini_im_1}

	if [ ! -f "${init_ini_fp}" ] && [ ! -f "${init_ini_im_1_fp}" ]; then
		echo "ERROR: both init_ini files ${init_ini} and ${init_ini_im_1} inaccessible at ${config_root}"
		exit 1
	fi

	if [ ! -f "${init_ini_fp}" ]; then
		if ! make_init_ini_file "${init_ini_im_1_fp}" "${init_ini_fp}" "${im_ratio}"; then
			echo "ERROR: failed to make \"${init_ini_fp}\" from \"${init_ini_im_1_fp}\""
			exit 1
		fi
	fi

	if ! set_reproduce "${init_ini_fp}" "${reproduce}"; then
		echo "ERROR: failed for set_reproduce \"${init_ini_fp}\" \"${reproduce}\""
		exit 1
	fi

	if [[ $n -eq ${n_first} ]]; then
		init_ckpt_d=${ckpt_d}
		if [ "${new_data}" == true ]; then
			if ! clean_data "${ckpt_d}"; then
				echo "ERROR: clean_data \"${ckpt_d}\": failed"
				exit 1
			fi
			echo "INFO: clean_data \"${ckpt_d}\": success"
		fi
		if [ ${rerun} == true ]; then
			echo "INFO: PYTHONPATH=methods/VCMatch/code/utils:uncertainty/src/ python uncertainty/src/hmsc_ps_vcmdata_shift_train.py -c ${init_ini_fp}"
			if ! PYTHONPATH=methods/VCMatch/code/utils:uncertainty/src/ python uncertainty/src/hmsc_ps_vcmdata_shift_train.py -c "${init_ini_fp}"; then
				echo "ERROR: hmsc_ps_vcmdata_shift_train.py failed"
				exit 1
			fi
		fi
	else
		sed -e "s/sigma = .*/sigma = ${sigma}/g" "${init_ini_fp}" > "${ini_fp}"
		echo "INFO: created configuration file '${ini_fp}'"
		if ! grep -q -E "${sigma}" "${ini_fp}"; then
			echo "incorrect sigma in ${ini_fp}"
			exit 1
		fi
		if [ ! -d "${ckpt_d}" ]; then
			rsync -avzr --exclude model_* "${init_ckpt_d}/" "${ckpt_d}"
			echo "INFO: ran rsync -avzr --exclude model_* '${init_ckpt_d}/' '${ckpt_d}'"
			if [ ! -f "${ini_fp}" ]; then
				echo "ERROR: configuration file ${ini_fp} inaccessible"
				exit 1
			fi
			echo "INFO: PYTHONPATH=methods/VCMatch/code/utils:uncertainty/src/ python uncertainty/src/hmsc_ps_vcmdata_shift_train.py -c ${ini_fp}"
			if ! PYTHONPATH=methods/VCMatch/code/utils:uncertainty/src/ python uncertainty/src/hmsc_ps_vcmdata_shift_train.py -c "${ini_fp}"; then
				echo "ERROR: hmsc_ps_vcmdata_shift_train.py failed"
				rm -rf "${ckpt_d}"
				exit 1
			fi

			if [ ! -d "${ckpt_d}" ]; then
				echo "ERROR: checkpoint at ${ckpt_d} still inaccessbile, hmsc_ps_vcmdata_shift_train.py failed"
				exit 1
			fi
		else
			if [ ${rerun} == true ]; then
				echo "INFO: PYTHONPATH=methods/VCMatch/code/utils:uncertainty/src/ python uncertainty/src/hmsc_ps_vcmdata_shift_train.py -c ${ini_fp}"
				if ! PYTHONPATH=methods/VCMatch/code/utils:uncertainty/src/ python uncertainty/src/hmsc_ps_vcmdata_shift_train.py -c "${ini_fp}"; then
					echo "ERROR: hmsc_ps_vcmdata_shift_train.py failed"
					exit 1
				fi
			else
				echo "Checkpoint ${ckpt_d} exists, skip ..."
			fi
		fi
		sleep 1
	fi
done

