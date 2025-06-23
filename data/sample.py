"""
用于混合不同领域的数据
"""
import os
import sys
import logging
import argparse
import copy
import json
import random

from os.path import join

from pdb import set_trace

logger = logging.getLogger(__name__)

def load_dataset(data_dir):
    dataset = []
    for fp in os.listdir(data_dir):
        with open(join(data_dir, fp), 'r') as f:
            for line in f:
                col = line.strip().split("\t")
                if len(col) == 2:
                    json_obj = json.loads(col[1])
                    dataset.append((col[0], json_obj["text"], json_obj["token_num"]))
  
    return dataset

def get_sample_dataset(dataset, token_num):
    sample_dataset = []

    random.shuffle(dataset)
    cur_token_num = 0
    for data in dataset:
        if cur_token_num >= token_num:
            break
        else:
            sample_dataset.append(data)
            cur_token_num += data[2]
    
    return sample_dataset, cur_token_num

if __name__ == "__main__":

    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S", level = logging.INFO
    )

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cn", type = float, default = 1/9)
    parser.add_argument("--baike", type = float, default = 1/9)
    parser.add_argument("--en", type = float, default = 1/9)
    parser.add_argument("--arxiv", type = float, default = 1/9)
    parser.add_argument("--math", type = float, default = 1/9)
    parser.add_argument("--code", type = float, default = 1/9)
    parser.add_argument("--instruction", type = float, default = 1/9)
    parser.add_argument("--ai_search", type = float, default = 1/9)
    parser.add_argument("--log_278", type = float, default = 1/9)
    parser.add_argument("--total_token_num", type = float, default = 1e9) # 10B
    parser.add_argument("--data_dir", type = str)
    parser.add_argument("--save_dir", type = str)

    args = parser.parse_args()

    token_num_sum = sum([args.cn, args.en, args.baike, args.arxiv, args.code, args.math, args.instruction, args.ai_search, args.log_278])
    token_num_dict = {
        "cn": int(args.cn / token_num_sum * args.total_token_num),
        "en": int(args.en / token_num_sum * args.total_token_num),
        "baike": int(args.baike / token_num_sum * args.total_token_num),
        "arxiv": int(args.arxiv / token_num_sum * args.total_token_num),
        "code": int(args.code / token_num_sum * args.total_token_num),
        "math": int(args.math / token_num_sum * args.total_token_num),
        "instruction": int(args.instruction / token_num_sum * args.total_token_num),
        "ai_search": int(args.ai_search / token_num_sum * args.total_token_num),
        "278": int(args.log_278 / token_num_sum * args.total_token_num)
    }

    for key in token_num_dict.keys():
        save_path = join(args.save_dir, f"{key}.jsonl")
        if not os.path.exists(save_path):
            dataset = load_dataset(join(args.data_dir, key))
            sample_dataset, cur_token_num = get_sample_dataset(dataset, token_num_dict[key])

            with open(save_path, "w", encoding = "utf-8") as f:
                for data in sample_dataset:
                    f.write(json.dumps({"text": data[1]}, ensure_ascii = False) + "\n")
            
            cur_token_num = cur_token_num / 1e9
            logger.info(f"data_type: {key}, token_num: {cur_token_num:.4f}B")