from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

def text_to_ids(tokenizer, batch):
    # When batched=True, batch['text'] is a list of strings
    return {'input_ids': [tokenizer.encode(text) for text in batch['text']]}


dataset = load_dataset('json',data_files='data/*.jsonl')

dataset = dataset.map(partial(text_to_ids,tokenizer),batched=True, num_proc=8)
        

print(dataset)