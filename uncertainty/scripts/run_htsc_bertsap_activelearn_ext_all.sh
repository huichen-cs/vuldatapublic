#!/bin/bash

if ! uncertainty/scripts/run_htsc_bertsap_activelearn_ext.sh 5 init; then
  exit 1
fi

if ! uncertainty/scripts/run_htsc_bertsap_activelearn_ext.sh 5 ehal_max; then
  exit 1
fi

if ! uncertainty/scripts/run_htsc_bertsap_activelearn_ext.sh 5 elah_max; then
  exit 1
fi

if ! uncertainty/scripts/run_htsc_bertsap_activelearn_ext.sh 5 random; then
  exit 1
fi

for ((e=1;e<=10;e++)); do
  uncertainty/scripts/run_htsc_bertsap_activelearn_ext.sh all eval | \
    tee run_htsc_bertsap_activelearn_ext_eval.log | \
    python uncertainty/src/save_result.py \
      htsc_bertsap_activelearn_ext_5_r11e$e.json
done
