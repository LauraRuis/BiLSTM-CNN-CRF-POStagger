#!/usr/bin/env bash

TAGGER_DIR="/home/joost/git/stanford-postagger-full-2015-12-09"

for DIR in "$HOME/data/wsj/wsj-conllx-3_3_0" "$HOME/data/wsj/wsj-conllx-3_5_0"; do
  for SPLIT in train dev.22 dev.24 test; do
    IN_PATH="${DIR}/${SPLIT}.conllx.form"
    OUT_PATH="${DIR}/${SPLIT}.conllx.form.tagged"
    java -cp "${TAGGER_DIR}/*:${TAGGER_DIR}/lib/*" edu.stanford.nlp.tagger.maxent.MaxentTagger -model ${TAGGER_DIR}/models/english-left3words-distsim.tagger -sentenceDelimiter newline -tokenize false -textFile ${IN_PATH} -outputFormat slashTags -outputFile ${OUT_PATH}
   done
done
