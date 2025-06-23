from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tiny-llama/")

def text_to_ids(tokenizer, batch):
    # When batched=True, batch['text'] is a list of strings
    try:
        res = {'input_ids': [tokenizer.encode(text) for text in batch['tex_source'] if 'text_source' in text]}
    except:
        print(batch)
    return res


dataset = load_dataset('json',data_files='data/arxiv/arXiv_src_23*.json')

dataset = dataset.map(partial(text_to_ids,tokenizer),batched=True, num_proc=8)
        

print(dataset)