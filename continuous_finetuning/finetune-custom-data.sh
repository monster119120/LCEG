#!/bin/bash

# ----------------- Scripts for origin Llama, PI, NTK and YaRN Methos-------------------

long_ratio=${1:-0.8}
token_num=${2:-5e8}
short_ratio=$((1 - long_ratio))

RECIPE_NAME=custom_data
METHOD_NAME=yarn # option:[origin, pi, ntk, yarn]
TRAINING_LENGTH=16384 
MODEL_PATH="../Llama-2-7b-hf/"
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}_long${long_ratio}_short${short_ratio}_token${token_num}

torchrun  --nproc_per_node=8 \
        fine-tune-custom-data.py  \
        --model_name_or_path ${MODEL_PATH} \
        --bf16 True \
        --output_dir ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --use_flash_attn True \
        --low_rank_training False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --eval_strategy "no" \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --lr_scheduler_type "constant_with_warmup" \
        --deepspeed  ds_configs/stage3.json \
        --logging_steps 100     \
        --tf32 True \
        --report_to "none" \
        --use_wandb False \
        --dataset_dir "/root/paddlejob/workspace/env_run/afs_data/kongrui/long_context_exp_data/long${long_ratio}_short${short_ratio}_token${token_num}/*.jsonl" \
        --method_name ${METHOD_NAME} \
        --wandb_name ${WANDB_NAME} 