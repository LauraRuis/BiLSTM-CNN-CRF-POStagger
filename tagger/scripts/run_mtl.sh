#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

bsz=2000
eval_every=1000
print_every=100

# baseline + mtl
#python -m parser --chinese --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --pos_emb_dim 0 --char_emb_dim 100 --recurrent_tagger --xpos --pos_loss_weight 0.01 --output_dir final_logs/final_mtl1_zh &> final_logs/log_mtl1_zh.txt
#python -m parser --chinese --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --pos_emb_dim 0 --char_emb_dim 100 --recurrent_tagger --xpos --pos_loss_weight 0.1 --output_dir final_logs/final_mtl2_zh &> final_logs/log_mtl2_zh.txt
python -m parser --chinese --batch_size $bsz --eval_every $eval_every --print_every $print_every --ud_2018 --plateau --pos_emb_dim 0 --char_emb_dim 100 --recurrent_tagger --xpos --pos_loss_weight 1 --output_dir final_logs/final_mtl3_zh &> final_logs/log_mtl3_zh.txt

