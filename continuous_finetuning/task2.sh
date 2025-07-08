#!/bin/bash

# ----------------- Scripts for origin Llama, PI, NTK and YaRN Methos-------------------

SOURCE_LEN=${1:-4096}
TARGET_LEN=${2:-32768}
MODEL_PATH="../Llama-2-7b-hf/"
CKPT_NAME=SOURCE_LEN_${SOURCE_LEN}_TARGET_LEN${TARGET_LEN}

MASTER_ADDR=localhost
MASTER_PORT=${2:-29501}
torchrun  --nproc_per_node=8 \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        fine-tune-custom-data.py  \
        --model_name_or_path ${MODEL_PATH} \
        --bf16 True \
        --output_dir ../ckpts/${CKPT_NAME} \
        --model_max_length ${TARGET_LEN} \
        --use_flash_attn True \
        --low_rank_training False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --eval_strategy "no" \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 1e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --lr_scheduler_type "constant_with_warmup" \
        --deepspeed  ds_configs/stage3_offload.json \
        --logging_steps 100     \
        --tf32 True \
        --report_to "tensorboard" \
        --use_wandb False \
        --dataset_dir "../long0.6_token5e8_no_select/*.jsonl" \
        --method_name yarn