#!/usr/bin/env bash
LOG_DIR="$HOME/Documents/AI/parser/logs_to_plot"  # put all log files you want to plot in this dir
ITER_COLUMN=8
LAS_COLUMN=10
UAS_COLUMN=12
XPOS_COLUMN=18
OUTPUT_FILE="${LOG_DIR}/plot"

INFILE1=$1
INFILE2=$2
if [[ -z "$1" ]]; then
    echo "please provide at least one path to a logfile (max 2)"
else

    function writefile {
        var=$1
        outfile_extension=${var##*/}
        file="${LOG_DIR}/${var}"


        if [ -e $file ]
        then
            # get file  | get rows with dev | get part with Iter | get first occuring integer | add Iter before it
            cat ${file} | grep -w "dev" | grep -oP '.*?\K Iter.*?(?=LAS|$)' | grep -Po '.*?\K[0-9]+' | awk '$0="Iter "$0' > "${OUTPUT_FILE}_${outfile_extension}";

            # get file | get rows with dev | grep part with metric | grep first occuring float | add metric before it
            cat ${file} | grep -w "dev" | grep -oP '.*?\K LAS.*?(?=UAS|$)' | grep -Po '^.*?\K\-?[0-9]+\.[0-9]+'  | awk '$0="LAS "$0' >> "${OUTPUT_FILE}_${outfile_extension}";
            cat ${file} | grep -w "dev" | grep -oP '.*?\K UAS.*?(?=MLAS|$)' | grep -Po '^.*?\K\-?[0-9]+\.[0-9]+'  | awk '$0="UAS "$0' >> "${OUTPUT_FILE}_${outfile_extension}";
            cat ${file} | grep -w "dev" | grep -oP '.*?\K POS.*?(?=UPOS|$)' | grep -Po '^.*?\K\-?[0-9]+\.[0-9]+'  | awk '$0="XPOS-ACC "$0' >> "${OUTPUT_FILE}_${outfile_extension}";
            cat ${file} | grep -w "dev" | grep -oP '.*?\K UPOS.*?(?=MORPH|$)' | grep -Po '^.*?\K\-?[0-9]+\.[0-9]+'  | awk '$0="UPOS-ACC "$0' >> "${OUTPUT_FILE}_${outfile_extension}";

        else
            echo "file not found"
            exit
        fi
    }
    writefile $1
    if [[ -z "$2" ]]; then
        outfile_extension1=${INFILE1##*/}
        file1="${LOG_DIR}/${INFILE1}"
        python scripts/plot_model.py "${OUTPUT_FILE}_${outfile_extension1}"
    else
        writefile $2
        outfile_extension1=${INFILE1##*/}
        file1="${LOG_DIR}/${INFILE1}"
        outfile_extension2=${INFILE2##*/}
        file2="${LOG_DIR}/${INFILE2}"
        python scripts/plot_model.py "${OUTPUT_FILE}_${outfile_extension1}" "${OUTPUT_FILE}_${outfile_extension2}"
    fi
fi