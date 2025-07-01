rsync -avh --partial --progress source/ destination/


--cn 1 --baike 1 --en 1 --arxiv 1 --math 2 --code 2 --instruction 2 --log_278 1


### 下载模型
```bash
rsync -avh --partial --progress workspace/env_run/afs_data/kongrui/Llama-2-7b-hf/ ./
```

### 准备数据
```bash
bash data/prepare_data.sh
```

### 开始训练
```bash
bash finetune-custom-data.sh 0.8 5e8
bash finetune-custom-data.sh 0.6 5e8
bash finetune-custom-data.sh 0.4 5e8
```

### 评估

```
./bcecmd bos cp -r bos:/digital-human-strategy/liqiyang/model_weight/longbench_data/ data/

```


## 从零开始运行机器

```bash
git clone https://github.com/monster119120/LCEG.git
cd LCEG/
pip install -r requirements-pdc.txt 

mkdir -p Llama-2-7b-hf
rsync -avh --partial --progress workspace/env_run/afs_data/kongrui/bcecmd ./
./bcecmd bos cp -r bos:/digital-human-strategy/liqiyang/model_weight/Llama-2-7b-hf/ Llama-2-7b-hf/

cp -r longbench/pred/llama2-7b-hf-slimpajama-yarn-16k-long0.8-token5e8/ workspace/env_run/afs_data/kongrui/
cp -r longbench/pred/llama2-7b-hf-slimpajama-yarn-16k-long0.4-token5e8/ workspace/env_run/afs_data/kongrui/

cp -r workspace/env_run/afs_data/kongrui/llama2-7b-hf-slimpajama-yarn-16k-long0.4-token5e8/ ./longbench/pred/
cp -r workspace/env_run/afs_data/kongrui/llama2-7b-hf-slimpajama-yarn-16k-long0.8-token5e8/ ./longbench/pred/
```


### 短文评估

lm_eval --model hf \
    --model_args pretrained=LCEG/continuous_finetuning/ckpts/custom_data/custom_data_pi_16384_long0.6_token5e8 \
    --tasks gsm8k \
    --device cuda:4 \
    --batch_size 8


lm_eval --model hf \
    --model_args pretrained=LCEG/Llama-2-7b-hf \
    --tasks mmlu \
    --device cuda:1 \
    --batch_size 8

lm_eval --model hf \
    --model_args pretrained=LCEG/LLaMa-2-7B-32K \
    --tasks mmlu \
    --device cuda:2 \
    --batch_size 8


lm_eval --model hf \
    --model_args pretrained=LCEG/continuous_finetuning/ckpts/custom_data/custom_data_yarn_16384_long0.6_token5e8 \
    --tasks hellaswag,mmlu,truthfulqa,arc_easy,arc_challenge,gsm8k \
    --device cuda:0 \
    --batch_size 8


lm_eval --model hf \
    --model_args pretrained=LCEG/Llama-2-7b-hf \
    --tasks hellaswag,mmlu,truthfulqa,arc_easy,arc_challenge,gsm8k \
    --device cuda:1 \
    --batch_size 8