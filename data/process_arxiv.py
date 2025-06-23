"""

long_context_data/arxiv_cleaned/目录下有很多.json文件，ls后如下所示
```
arXiv_src_2309_131_cleaned.json
arXiv_src_2409_131_cleaned.json
arXiv_src_2509_131_cleaned.json
```

每个json文件里的格式如下
```
[
  {
    "id": "2301.00001",
    "submitter": "Tauheed Khan Mohd",
    "authors": "Jordan Thompson, Ryan Benac, Kidus Olana, Talha Hassan, Andrew Sward, Tauheed Khan Mohd",
    "title": "NFTrig",
    "comments": null,
    "journal-ref": null,
    "doi": null,
    "abstract": "NFTrig is a web-based application created for use as an educational tool to teach trigonometry and block chain technology. Creation of the application includes front and back end development as well as integration with other outside sources including MetaMask and OpenSea. The primary development languages include HTML, CSS (Bootstrap 5), and JavaScript as well as Solidity for smart contract creation. The application itself is on Moralis utilizing their Web3 API. This technical report describes how the application was created, what the application requires, and smart contract design with security considerations in mind. The NFTrig application has underwent significant testing and validation prior to and after deployment. Future suggestions and recommendations for further development, maintenance, and use in other fields for education are also described.",
    "report-no": null,
    "categories": [
      "cs.HC"
    ],
    "versions": [
      "v1"
    ],
    "versions_dates": [
      "Wed, 21 Dec 2022 18:07:06 GMT"
    ],
    "tex_source": "xx..."
  },
  {
...
```

这个是我的tokenizer
```
tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-hf/")
input_ids = tokenizer.encode(tex_source)
```

我希望把所有的 input_ids 长度大于 16384 的 tex_source 保存到 long_context_processed_data/arxiv/ 目录下，每条数据包含tex_source

"""
import os
import json
import glob
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_DIR = "long_context_data/arxiv_cleaned/"
OUTPUT_DIR = "long_context_processed_data/arxiv/"
TOKENIZER_NAME = "Llama-2-7b-hf/"
MIN_LENGTH = 16384
MAX_DOCS_PER_FILE = 1000
NUM_THREADS = 20

def process_file(file_path, tokenizer):
    """
    Reads a single JSON file, tokenizes the 'tex_source' of each article,
    and returns a list of sources that are longer than MIN_LENGTH.
    """
    long_articles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
        
        for article in data:
            tex_source = article.get("tex_source")
            if isinstance(tex_source, str) and tex_source.strip():
                input_ids = tokenizer.encode(tex_source, add_special_tokens=False)
                if len(input_ids) > MIN_LENGTH:
                    long_articles.append(tex_source)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Skipping file {file_path} due to error: {e}")
    return long_articles

def process_arxiv_data():
    """
    Processes arXiv JSON files to find long TeX sources using multiple threads.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    file_paths = glob.glob(os.path.join(INPUT_DIR, "*_cleaned.json"))
    if not file_paths:
        print(f"No '_cleaned.json' files found in {INPUT_DIR}")
        return

    print(f"Found {len(file_paths)} files to process with {NUM_THREADS} threads. Output will be saved in chunks of {MAX_DOCS_PER_FILE} to {OUTPUT_DIR}")

    all_long_sources = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_path = {executor.submit(process_file, path, tokenizer): path for path in file_paths}
        
        for future in tqdm(as_completed(future_to_path), total=len(file_paths), desc="Processing files"):
            long_sources_from_file = future.result()
            all_long_sources.extend(long_sources_from_file)

    print(f"\nWriting {len(all_long_sources)} documents to output files...")
    
    long_doc_count = 0
    file_index = 0
    f_out = None
    try:
        for tex_source in all_long_sources:
            if long_doc_count % MAX_DOCS_PER_FILE == 0:
                if f_out:
                    f_out.close()
                output_file_path = os.path.join(OUTPUT_DIR, f"arxiv_long_context_{file_index}.jsonl")
                f_out = open(output_file_path, 'w', encoding='utf-8')
                file_index += 1
            
            long_article_data = {"tex_source": tex_source}
            if f_out:
                f_out.write(json.dumps(long_article_data) + '\n')
            long_doc_count += 1
    finally:
        if f_out:
            f_out.close()
    
    print(f"\nProcessing complete.")
    print(f"Found {long_doc_count} documents with token length > {MIN_LENGTH}.")
    print(f"Results saved across {file_index} file(s) in {OUTPUT_DIR}")

if __name__ == "__main__":
    process_arxiv_data()
