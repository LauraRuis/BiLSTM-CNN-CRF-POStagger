#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

bsz=2000
eval_every=1000
print_every=100
train_path=data/ud/UD_English-wst/2018/tagged/en-ud-train.conllu

# baseline
#python -m parser --train_path $train_path --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --xpos_input --output_dir final_logs/final_baseline1-tagged &> final_logs/log_baseline1-tagged.txt
python -m parser --train_path $train_path --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --xpos_input --char_emb_dim 100 --output_dir final_logs/final_baseline2-tagged &> final_logs/log_baseline2-tagged.txt
python -m parser --train_path $train_path --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --xpos_input --word_dropout_p 0 --pos_dropout_p 0 --output_dir final_logs/final_baseline3-tagged &> final_logs/log_baseline3-tagged.txt
python -m parser --train_path $train_path --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --xpos_input --char_emb_dim 100 --simple_char_model --output_dir final_logs/final_baseline4-tagged &> final_logs/log_baseline4-tagged.txt
