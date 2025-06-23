#!/bin/bash

python data/sample_short.py \
    --data_dir "data_pool/" \
    --save_dir "data_pool/sampled_data" \
    --cn 0.01 --baike 0.01 --en 0.01 --arxiv 0.01 --math 0.02 --code 0.02 --instruction 0.02 --log_278 0.01 \
    --ratio 0.4 \
    --total_token_num 9e8   # 0.9B


python data/sample_long.py \
    --data_dir "data_pool/" \
    --save_dir "data_pool/sampled_data" \
    --cn 0.01 --baike 0.01 --en 0.01 --arxiv 0.01 --math 0.02 --code 0.02 --instruction 0.02 --log_278 0.01 \
    --ratio 0.6 \
    --total_token_num 9e8   # 0.9B


# ----------------- Scripts for origin Llama, PI, NTK and YaRN Methos-------------------
RECIPE_NAME=custom_data
METHOD_NAME=pi # option:[origin, pi, ntk, yarn]
TRAINING_LENGTH=16384 
MODEL_PATH="../Llama-2-7b-hf/"
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}

torchrun  --nproc_per_node=8 \
        fine-tune-custom-data.py  \
        --model_name_or_path ${MODEL_PATH} \
        --bf16 True \
        --output_dir ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --use_flash_attn True \
        --low_rank_training False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --eval_strategy "no" \
        --save_strategy "epoch" \
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
        --dataset_dir "../data_pool/sampled_data/*.jsonl" \
        --method_name ${METHOD_NAME} \
        --wandb_name ${WANDB_NAME} 