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
from multiprocessing import Pool, cpu_count

from os.path import join
from tqdm import tqdm

from pdb import set_trace

logger = logging.getLogger(__name__)

def _process_file(file_path):
    """Helper function to process a single file."""
    dataset_part = []
    try:
        # It's good practice to specify encoding
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                col = line.strip().split("\t")
                if len(col) == 2:
                    try:
                        json_obj = json.loads(col[1])
                        # Check for keys to avoid KeyError
                        if 'text' in json_obj and 'token_num' in json_obj:
                            dataset_part.append((col[0], json_obj["text"], json_obj["token_num"]))
                        else:
                            # This might be too noisy for large datasets
                            # logger.warning(f"Skipping line in {file_path} due to missing 'text' or 'token_num' key.")
                            pass
                    except json.JSONDecodeError:
                        # This might be too noisy as well.
                        # logger.warning(f"Skipping line in {file_path} due to JSON decode error.")
                        pass
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
    return dataset_part

def load_dataset(data_dir):
    try:
        filepaths = [join(data_dir, fp) for fp in os.listdir(data_dir)]
        filepaths = [f for f in filepaths if os.path.isfile(f)]
    except FileNotFoundError:
        logger.error(f"Data directory not found: {data_dir}")
        return []
    
    if not filepaths:
        logger.warning(f"No files found in directory: {data_dir}")
        return []

    dataset = []
    num_processes = 30
    
    with Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(_process_file, filepaths)
        
        for result in tqdm(results, total=len(filepaths), desc=f"Loading files from {data_dir}"):
            dataset.extend(result)
                
    return dataset

def get_sample_dataset(dataset, token_num):
    sample_dataset = []

    random.shuffle(dataset)
    cur_token_num = 0
    for data in dataset:
        if token_num >= 16384:
            continue

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
    parser.add_argument("--total_token_num", type = float, default = 1e9) # 1B
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

    token_num_dict = {
        "278":  args.total_token_num
    }

    for key in token_num_dict.keys():
        os.makedirs(args.save_dir, exist_ok = True)
        save_path = join(args.save_dir, f"{key}.jsonl")
        
        dataset = load_dataset(join(args.data_dir, key))
        print(f"Loaded {len(dataset)} documents from {key}")

        sample_dataset, cur_token_num = get_sample_dataset(dataset, token_num_dict[key])

        with open(save_path, "w", encoding = "utf-8") as f:
            for data in tqdm(sample_dataset, desc=f"Writing {key} data"):
                f.write(json.dumps({"text": data[1]}, ensure_ascii = False) + "\n")
        
        cur_token_num = cur_token_num / 1e9
        logger.info(f"data_type: {key}, token_num: {cur_token_num:.4f}B")