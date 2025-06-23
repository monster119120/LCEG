from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tiny-llama/")

def text_to_ids(tokenizer, batch):
    # When batched=True, batch['text'] is a list of strings
    return {'input_ids': [tokenizer.encode(text) for text in batch['tex_source']]}


dataset = load_dataset('json',data_files='data/code/processed_arxiv_cleaned/arXiv_src_2505_116_cleaned.json')

dataset = dataset.map(partial(text_to_ids,tokenizer),batched=True, num_proc=8)
        

print(dataset)