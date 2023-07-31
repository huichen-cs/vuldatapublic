#!/bin/bash

if [[ $# -ne 3 ]]; then
	echo "Usage: $0 im_ratio task_begin_no task_end_no"
	exit 0
fi


# midwood
# im_ratio=1
# task_begin=20
# task_end=30

# wang
# im_ratio=3
# task_begin=10
# task_end=20
im_ratio=$1
task_begin=$2
task_end=$3
for ((i=task_begin; i<task_end; i++)); do

	echo "i = $i im_ratio = ${im_ratio} tasks = ${task_begin} ${task_end}"

	# Train
	time (uncertainty/scripts/run_psvcmatch_hmsc_gn_shift.sh \
		--n_first 0 --n_last 40 --im_ratio "${im_ratio}" --rerun |  \
		tee "en_si_0_40_im_${im_ratio}_r_${i}_v5.log" | \
		python uncertainty/src/save_result.py \
		"en_si_0_40_im_${im_ratio}_r_${i}_v5.json")

	# Eval - ensemble, dropout, vanilla
	#        dropout/vanilla: select member model with best_f1, median_f1, or random
	time (uncertainty/scripts/run_psvcmatch_hmsc_gn_shift_eval_only.sh \
			--n_first 0 --n_last 40 --im_ratio  "${im_ratio}" \
			--uq_method ensemble | \
		tee en_si_0_40_im_"${im_ratio}"_r_${i}_v5.log | \
		python uncertainty/src/save_result.py \
			en_si_0_40_im_"${im_ratio}"_r_${i}_v5.json)

	time (uncertainty/scripts/run_psvcmatch_hmsc_gn_shift_eval_only.sh \
			--n_first 0 --n_last 40 --im_ratio  "${im_ratio}" \
			--uq_method vanilla --select_criteria best_f1 | \
		tee va_si_0_40_im_"${im_ratio}"_r_${i}_v5.log | \
		python uncertainty/src/save_result.py \
			va_si_0_40_im_"${im_ratio}"_r_${i}_v5.json)

	time (uncertainty/scripts/run_psvcmatch_hmsc_gn_shift_eval_only.sh \
			--n_first 0 --n_last 40 --im_ratio  "${im_ratio}" \
			--uq_method vanilla --select_criteria median_f1 | \
		tee va_si_0_40_im_"${im_ratio}"_r_${i}_v5_median.log | \
		python uncertainty/src/save_result.py \
			va_si_0_40_im_"${im_ratio}"_r_${i}_v5_median.json)

	time (uncertainty/scripts/run_psvcmatch_hmsc_gn_shift_eval_only.sh \
			--n_first 0 --n_last 40 --im_ratio  "${im_ratio}" \
			--uq_method vanilla --select_criteria random | \
		tee va_si_0_40_im_"${im_ratio}"_r_${i}_v5_random.log | \
		python uncertainty/src/save_result.py \
			va_si_0_40_im_"${im_ratio}"_r_${i}_v5_random.json)

	time (uncertainty/scripts/run_psvcmatch_hmsc_gn_shift_eval_only.sh \
			--n_first 0 --n_last 40 --im_ratio "${im_ratio}" \
			--uq_method dropout --select_criteria best_f1 | \
		tee do_si_0_40_im_"${im_ratio}"_r_${i}_v5.log | \
		python uncertainty/src/save_result.py \
			do_si_0_40_im_"${im_ratio}"_r_${i}_v5.json)

	time (uncertainty/scripts/run_psvcmatch_hmsc_gn_shift_eval_only.sh \
			--n_first 0 --n_last 40 --im_ratio "${im_ratio}" \
			--uq_method dropout --select_criteria median_f1 | \
		tee do_si_0_40_im_"${im_ratio}"_r_${i}_v5_median.log | \
		python uncertainty/src/save_result.py \
			do_si_0_40_im_"${im_ratio}"_r_${i}_v5_median.json)

	time (uncertainty/scripts/run_psvcmatch_hmsc_gn_shift_eval_only.sh \
			--n_first 0 --n_last 40 --im_ratio "${im_ratio}" \
			--uq_method dropout --select_criteria random | \
		tee do_si_0_40_im_"${im_ratio}"_r_${i}_v5_random.log | \
		python uncertainty/src/save_result.py \
			do_si_0_40_im_"${im_ratio}"_r_${i}_v5_random.json)
done

