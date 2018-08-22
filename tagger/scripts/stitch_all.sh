#!/usr/bin/env bash

for DIR in "$HOME/data/wsj/wsj-conllx-3_3_0" "$HOME/data/wsj/wsj-conllx-3_5_0"; do
  for SPLIT in train dev test; do
    echo $DIR $SPLIT
    python3 -m parser.tools.stitch ${DIR}/${SPLIT}.conllx ${DIR}/notext.${SPLIT}.stanford.conll > ${DIR}/${SPLIT}.stanford.conll
   done
done
