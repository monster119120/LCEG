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
from tqdm import tqdm

from pdb import set_trace

logger = logging.getLogger(__name__)

def stream_dataset(data_dir):
    """
    Streams dataset from a directory, yielding one item at a time.
    """
    for fp in os.listdir(data_dir):
        with open(join(data_dir, fp), 'r') as f:
            for line in f:
                col = line.strip().split("\t")
                if len(col) == 2:
                    try:
                        json_obj = json.loads(col[1])
                        if "text" in json_obj and "token_num" in json_obj:
                            yield (col[0], json_obj["text"], json_obj["token_num"])
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line in {fp}")


def get_sample_dataset(data_dir, token_num):
    """
    Samples a dataset from a directory using reservoir sampling.
    The method involves two passes over the data, but avoids loading it all into memory.
    """
    # 1. First pass: count total documents and total tokens
    total_docs = 0
    total_tokens = 0
    desc = f"Pass 1/2: Counting in {os.path.basename(data_dir)}"
    for _, _, item_token_num in tqdm(stream_dataset(data_dir), desc=desc):
        total_docs += 1
        total_tokens += item_token_num
    
    if total_docs == 0:
        return [], 0

    avg_tokens_per_doc = total_tokens / total_docs
    
    if avg_tokens_per_doc == 0:
        return [], 0

    # 2. Estimate number of samples (k) for the reservoir
    k = int(token_num / avg_tokens_per_doc)
    if k == 0:
        return [], 0
    
    k = min(k, total_docs) # Cannot sample more than available

    # 3. Second pass: perform reservoir sampling
    reservoir = []
    docs_seen = 0
    cur_token_num_in_reservoir = 0
    
    desc = f"Pass 2/2: Sampling from {os.path.basename(data_dir)}"
    with tqdm(total=total_docs, desc=desc) as pbar:
        for data in stream_dataset(data_dir):
            docs_seen += 1
            if len(reservoir) < k:
                reservoir.append(data)
                cur_token_num_in_reservoir += data[2]
            else:
                j = random.randint(0, docs_seen - 1)
                if j < k:
                    cur_token_num_in_reservoir -= reservoir[j][2]
                    reservoir[j] = data
                    cur_token_num_in_reservoir += data[2]
            
            pbar.update(1)
            pbar.set_postfix_str(f"Sampled Tokens: {cur_token_num_in_reservoir/1e9:.4f}B")

    return reservoir, cur_token_num_in_reservoir

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
    parser.add_argument("--total_token_num", type = float, default = 1e6) # 10B
    parser.add_argument("--data_dir", type = str, default = "data_pool/")
    parser.add_argument("--save_dir", type = str, default = "tmp/")

    args = parser.parse_args()

    token_num_sum = sum([args.cn, args.en, args.baike, args.arxiv, args.code, args.math, args.instruction, args.ai_search, args.log_278])
    # token_num_dict = {
    #     "cn": int(args.cn / token_num_sum * args.total_token_num),
    #     "en": int(args.en / token_num_sum * args.total_token_num),
    #     "baike": int(args.baike / token_num_sum * args.total_token_num),
    #     "arxiv": int(args.arxiv / token_num_sum * args.total_token_num),
    #     "code": int(args.code / token_num_sum * args.total_token_num),
    #     "math": int(args.math / token_num_sum * args.total_token_num),
    #     "instruction": int(args.instruction / token_num_sum * args.total_token_num),
    #     "ai_search": int(args.ai_search / token_num_sum * args.total_token_num),
    #     "278": int(args.log_278 / token_num_sum * args.total_token_num)
    # }

    # For debug
    token_num_dict = {
        "278": args.total_token_num,
    }

    for key in token_num_dict.keys():
        os.makedirs(args.save_dir, exist_ok = True)
        save_path = join(args.save_dir, f"{key}.jsonl")
        if not os.path.exists(save_path):
            sample_dataset, cur_token_num = get_sample_dataset(join(args.data_dir, key), token_num_dict[key])

            with open(save_path, "w", encoding = "utf-8") as f:
                for data in sample_dataset:
                    f.write(json.dumps({"text": data[1]}, ensure_ascii = False) + "\n")
            
            cur_token_num = cur_token_num / 1e9
            logger.info(f"data_type: {key}, token_num: {cur_token_num:.4f}B")