#!/bin/bash

cat 0[2-9]/wsj_* 1*/wsj_* 20/wsj_* 21/wsj_* > train.conllx
cat 24/wsj_* > dev.24.conllx
cat 23/wsj_* > test.conllx
cat 22/wsj_* > dev.22.conllx

cat train.conllx | egrep "^1[[:space:]]+" | wc -l
cat dev.22.conllx | egrep "^1[[:space:]]+" | wc -l
cat dev.24.conllx | egrep "^1[[:space:]]+" | wc -l
cat test.conllx | egrep "^1[[:space:]]+" | wc -l
