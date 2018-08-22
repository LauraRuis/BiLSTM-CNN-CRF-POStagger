#!/usr/bin/env bash

awk 'FNR==NR{a[NR]=$5;next}{$5=a[FNR]}1' final_logs/final_mtl3_zh/dev.iter00059000.conll data/ud/UD_Chinese-GSD/zh-ud-dev.conllu &> test.txt
tr ' ' '\t' < test.txt<> test.txt &> test2.txt