#!/usr/bin/env bash
LOG_DIR="$HOME/Documents/AI/parser/logs_to_plot"  # put all log files you want to plot in this dir
ITER_COLUMN=8
LAS_COLUMN=10
UAS_COLUMN=12
XPOS_COLUMN=18
OUTPUT_FILE="${LOG_DIR}/plot"

for dir in ${LOG_DIR}
do
    for file in ${dir}/*.txt; do

        outfile_extension=${file##*/}
        # continue if log file
        [[ $outfile_extension == log* ]] || continue
            # echo "Current File: ${file}"
            # echo "${outfile_extension}"
            # echo LAS UAS XPOS > "${OUTPUT_FILE}_${outfile_extension}"

            # get file  | get rows with dev | get part with Iter | get first occuring integer | add Iter before it
            cat ${file} | grep -w "dev" | grep -oP '.*?\K Iter.*?(?=LAS|$)' | grep -Po '.*?\K[0-9]+' | awk '$0="Iter "$0' > "${OUTPUT_FILE}_${outfile_extension}";

            # get file | get rows with dev | grep part with metric | grep first occuring float | add metric before it
            cat ${file} | grep -w "dev" | grep -oP '.*?\K LAS.*?(?=UAS|$)' | grep -Po '^.*?\K\-?[0-9]+\.[0-9]+'  | awk '$0="LAS "$0' >> "${OUTPUT_FILE}_${outfile_extension}";
            cat ${file} | grep -w "dev" | grep -oP '.*?\K UAS.*?(?=MLAS|$)' | grep -Po '^.*?\K\-?[0-9]+\.[0-9]+'  | awk '$0="UAS "$0' >> "${OUTPUT_FILE}_${outfile_extension}";
            cat ${file} | grep -w "dev" | grep -oP '.*?\K POS.*?(?=UPOS|$)' | grep -Po '^.*?\K\-?[0-9]+\.[0-9]+'  | awk '$0="XPOS-ACC "$0' >> "${OUTPUT_FILE}_${outfile_extension}";
            cat ${file} | grep -w "dev" | grep -oP '.*?\K UPOS.*?(?=MORPH|$)' | grep -Po '^.*?\K\-?[0-9]+\.[0-9]+'  | awk '$0="UPOS-ACC "$0' >> "${OUTPUT_FILE}_${outfile_extension}";
    done
    python3 scripts/plot_scores.py
done