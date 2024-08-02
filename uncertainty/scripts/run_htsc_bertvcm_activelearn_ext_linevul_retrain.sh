#!/bin/bash

###
# Example to run this script
# 1. To run the initial training with 20% of the data (1/20% = 5)
#    uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh 5 init
# 2. To run the active learning with ehal (entropy-based heuristic active learning) - best case
#    uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh 5 ehal_max
# 3. To run the active learning with ehal (entropy-based heuristic active learning) - worst case
#    uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh 5 elah_max
# 4. To run the active learning with random
#    uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh 5 random
# 5. To evaluate the models
#    uncertainty/scripts/run_htsc_bertvcm_activelearn_ext.sh all eval
###

###
# to run all pairs as follows:
###
# PYTHONPATH=uncertainty/src \
#   python uncertainty/src/htsc_linevul_activelearn_data_select.py \
#     -c uncertainty/config/htscbertvcm/active/en_bertvcm_1.0_im_1_train_0.9_sigma_0_ext.ini \
#     -a ehal_max \
#     --linevul_input_file data/linevul/linevul_train_vul.csv \
#     --output_file_dir output/linevul/linevul_train_vul_ehal_max.csv \
#   ...

run_config_file="uncertainty/config/htscbertvcm/active/en_bertvcm_1.0_im_1_train_0.9_sigma_0_ext.ini"

if [[ $# -lt 4 ]]; then
	echo "Usage: $0 mode_index active_learn_method data_selection_ratio runid"
	echo "Usage: $0 mode_index active_learn_method data_selection_ratio runid batchsize"
	exit 0
fi

[ -f .env ] || (echo "unable to access from $(pwd)/.env" && exit 1)

. .env

if [[ $# -ge 4 ]]; then
    model=$1
	method=$2
    infer_batch_size=$3
	runid=$4
fi

if [[ $# -ge 5 ]]; then
	batchsize=$5
fi


linevul_input_file="data/linevul/linevul_train_vul.csv"
linevul_output_dir="output/linevul/retrain"

if [[ $# -ge 6 ]]; then
  linevul_input_file=$6
  base_fn=$(basename "${linevul_input_file}")
  base_fn=${base_fn/.csv/}
  if [[ "${linevul_input_file}" == *"/splits/"* ]]; then
    linevul_output_dir="output/linevul/commit/retrain/"
  fi
fi

if [[ ! -f "${linevul_input_file}" ]]; then
  echo "linevul_input_file ${linevul_input_file} inaccessible!"
  exit 1
fi

if [ -n "${runid}" ]; then
	. uncertainty/scripts/luxury_setup.sh
	runidruncnf=$(mk_config_file_for_runid "${run_config_file}" "${runid}")
	if [ -z "${runidruncnf}" ]; then
		echo "failed to obtain new run configuration file for runid '${runid}'"
		exit 1
	fi
	if [ ! -f "${runidruncnf}" ]; then
		echo "failed to create run configuration for runid '${runid}' at '${runidruncnf}'"
		exit 1
	fi
	run_config_file="${runidruncnf}"

	if [ -n "${batchsize}" ]; then
		update_batch_size "${run_config_file}" "${batchsize}"
	fi
fi



mkdir -p "output/linevul"
echo "TOKENIZERS_PARALLELISM=false PYTHONPATH=$PYTHONPATH \\"
echo "  python uncertainty/src/htsc_linevul_activelearn_data_select_retrain.py \\"
echo "    -c ${run_config_file} \\"
echo "    -a \"${method}\" \\"
echo "    --linevul_input_file \"${linevul_input_file}\" \\"
echo "    --output_file_dir \"${linevul_output_dir}\" \\"
echo "    --infer_batch_size \"${infer_batch_size}\" \\"
echo "    --model_index \"${model}\""
TOKENIZERS_PARALLELISM=false PYTHONPATH=$PYTHONPATH \
    python uncertainty/src/htsc_linevul_activelearn_data_select_retrain.py \
    -c "${run_config_file}" \
    -a "${method}" \
    --linevul_input_file "${linevul_input_file}" \
    --output_file_dir "${linevul_output_dir}" \
    --infer_batch_size "${infer_batch_size}" \
    --model_index "${model}"

