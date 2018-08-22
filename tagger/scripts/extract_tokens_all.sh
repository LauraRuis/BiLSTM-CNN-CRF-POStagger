#!/usr/bin/env bash

for DIR in "$HOME/data/wsj/wsj-conllx-3_3_0" "$HOME/data/wsj/wsj-conllx-3_5_0"; do
  for SPLIT in train dev.22 dev.24 test; do
    python3 -m parser.tools.extract_tokens ${DIR}/${SPLIT}.conllx > ${DIR}/${SPLIT}.conllx.form
   done
done
