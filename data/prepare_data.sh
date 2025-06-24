

for long_ratio in 0.4 0.6 0.8; do
    for token_num in 5e8 1e9 5e9; do

        save_dir="/root/paddlejob/workspace/env_run/afs_data/kongrui/long_context_exp_data/long${long_ratio}_short${1-long_ratio}_token${token_num}"

        python data/sample_short.py \
            --data_dir "data_pool/" \
            --save_dir $save_dir \
            --cn 2.875 --baike 2.875 --en 2.875 --arxiv 2.875 --math 19 --code 8 --instruction 8.05 --log_278 1.725 --ai_search 1.725 \
            --ratio $((1 - long_ratio)) \
            --total_token_num $token_num

        python data/sample_long.py \
            --data_dir "data_pool/" \
            --save_dir $save_dir \
            --cn 2.875 --baike 2.875 --en 2.875 --arxiv 2.875 --math 19 --code 8 --instruction 8.05 --log_278 1.725 --ai_search 1.725 \
            --ratio $long_ratio \
            --total_token_num $token_num

    done
done