#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

bsz=2000
eval_every=1000
print_every=100
train_path=data/ud/UD_English-wst/2018/tagged/en-ud-train-ownpos.conllu

# baseline
python -m parser --chinese --train_path $train_path --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --xpos_input --output_dir final_logs/final_baseline1-tagged_zh_extra &> final_logs/log_baseline1-tagged_zh_extra.txt
python -m parser --train_path $train_path --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --xpos_input --output_dir final_logs/final_baseline1-tagged_extra &> final_logs/log_baseline1-tagged_extra.txt
