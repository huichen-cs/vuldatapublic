#!/bin/bash

function mk_config_file_for_runid() {
    cnf=$1
    runid=$2
    extension="${cnf##*.}"
    filename="${cnf%.*}"
    runcnf=${filename}_${runid}.${extension}

    if ! cp ${cnf} ${runcnf}; then
        echo "failed to 'cp ${cnf} ${runcnf}'"
        exit 1
    fi
    if ! grep -q "^dir_path = " "${runcnf}"; then
        echo "failed to locate '^dir_path = ' in '${runcnf}'"
        exit 1
    fi

    if ! sed --in-place=.bu -e "s/^\(dir_path = .*$\)/\1_${runid}/g" "${runcnf}"; then
        echo "failed to update '${runcnf}'"
        exit 1
    fi
    
    echo ${runcnf}
}

# update_batch_size "${run_config_file}" "${batchsize}"
function update_batch_size() {
    runcnf=$1
    batchsize=$2
    if ! sed --in-place=.bs.bu -e "s/^batch_size = .*$/batch_size = ${batchsize}/g" "${runcnf}"; then
        echo "failed to update '${runcnf}'"
        exit 0
    fi
}

# runidruncnf=$(mk_config_file_for_runid "${run_config_file}" "${runid}")
# echo ${runidruncnf}