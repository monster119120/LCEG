### longbench 
datasets=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" \
          "gov_report" "qmsum" "multi_news" "trec" "triviaqa" "samsum" \
          "passage_count" "passage_retrieval_en" "lcc" "repobench-p")

### many shots trec
# datasets=("trec_1000shots" "trec_875shots" "trec_750shots" "trec_625shots" "trec_500shots" "trec_400shots" "trec_300shots" "trec_200shots" "trec_100shots" "trec_75shots" "trec_50shots" "trec_25shots" "trec_10shots" "trec_5shots" "trec_1shots")


### longbench-e
# datasets=( "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" \
#           "gov_report" "multi_news" "trec" "triviaqa" "samsum" \
#           "passage_count" "passage_retrieval_en" "lcc" "repobench-p")

### models
models=(
    # "llama2-7b-hf" \
    # "llama2-7b-hf-slimpajama-landmark" \
    # "llama2-7b-hf-lminfinite" \
    # "llama2-7b-hf-slimpajama-pi-32k" \
    # "llama2-7b-hf-slimpajama-longlora-32k" \ 
    # "llama2-7b-hf-ntk-frozen" \
    # "llama2-7b-hf-slimpajama-ntk-32k" \
    # "llama2-7b-hf-slimpajama-ntk-64k" \
    # "llama-2-7b-hf-slimpajama-ntk-64k-2B" \
    # "llama2-7b-hf-slimpajama-yarn-32k"
    # "llama2-7b-hf-slimpajama-yarn-16k-long0.8-short0.2" \
    # "llama2-7b-hf-slimpajama-yarn-16k-long0.6-short0.4" \
    "llama2-7b-hf-slimpajama-pi-16k-long0.4-token5e8" \
    "llama2-7b-hf-slimpajama-pi-16k-long0.6-token5e8" \
    "llama2-7b-hf-slimpajama-pi-16k-long0.8-token5e8" \
    # "llama2-7b-hf-slimpajama-yarn-16k-long0.6-token5e8" \
    # "llama2-7b-hf-slimpajama-yarn-16k-long0.4-token5e8" \
    # "llama2-7b-hf-slimpajama-yarn-16k-long0.8-token5e8" \
    )

### models test 4k
# models=(
#     "llama2-7b-hf-slimpajama-landmark-test4k" \
#     "llama2-7b-hf-lminfinite-test4k" \
#     "llama2-7b-hf-slimpajama-pi-32k-test4k" \
#     "llama2-7b-hf-slimpajama-longlora-32k-test4k"  \
#     "llama2-7b-hf-ntk-frozen-test4k" \
#     "llama2-7b-hf-slimpajama-ntk-64k-test4k" \
#     "llama-2-7b-hf-slimpajama-ntk-64k-2B-test4k" \
#     "llama2-7b-hf-slimpajama-yarn-32k-test4k"
# )

# Get the total number of datasets
num_datasets=${#datasets[@]}
# Define the GPUs to use
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}

# Process datasets in batches of 8, running each dataset on a separate GPU
for ((i=0; i<num_datasets; i+=num_gpus)); do
    for j in $(seq 0 $((num_gpus - 1))); do
        dataset_index=$((i + j))
        # Ensure we don't go past the end of the datasets array
        if [ $dataset_index -lt $num_datasets ]; then
            dataset=${datasets[$dataset_index]}
            gpu_id=${gpus[$j]}
            (
                for MODEL_NAME in "${models[@]}"; do
                    echo "Starting dataset ${dataset} with model ${MODEL_NAME} on GPU ${gpu_id}"
                    CUDA_VISIBLE_DEVICES=$gpu_id python pred.py \
                        --model "${MODEL_NAME}" \
                        --dataset_name "${dataset}"
                done
            ) &
        fi
    done
    # Wait for all background jobs in the current batch to complete
    wait
done

