#!/bin/bash

# ----------------- Scripts for origin Llama, PI, NTK and YaRN Methos-------------------

long_ratio=${1:-0.8}

RECIPE_NAME=custom_data
TRAINING_LENGTH=16384 
MODEL_PATH="./Llama-2-7b-hf/"
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}_long${long_ratio}_token${token_num}

MASTER_ADDR=localhost
MASTER_PORT=${4:-29501}
torchrun  --nproc_per_node=8 \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
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
        --dataset_dir "long0.6_token5e8/*.jsonl" \
        --method_name yarn \
        --wandb_name ${WANDB_NAME} 