from typing import Dict, List
from collections import defaultdict
from transformers import AutoTokenizer
import argparse
import datasets

import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)


def tokenize(
    batch,
    tokenizer,
    key: str = "text",
) -> Dict[str, List]:
    texts = batch[key]
    features = defaultdict(list)
    for text in texts:
        tokenized_text = tokenizer.encode(text)
        features[f"tokenized_{key}"].append(tokenized_text)
        features["n_tokens"].append(len(tokenized_text))
    return features 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=str, required=True, help="Path of HF dataset")
    parser.add_argument("--hf-dir", type=str, default=None, help="Dir in HF dataset")
    parser.add_argument("--hf-tokenizer", type=str, required=True, help="Path of HF tokenizer")
    parser.add_argument("--key", type=str, default='text', help="Name of the column that contains text to tokenize")
    parser.add_argument("--save-path", type=str, help="Folder to save processed HF dataset to")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of processes for HF processing")
    args = parser.parse_args()

    logging.info("Loading the dataset")
    ds = datasets.load_dataset(
        path=args.hf_path,
        data_dir=args.hf_dir,
        num_proc=args.num_proc,
        split="train",
        trust_remote_code=True
    )
    
    logging.info("Loading the tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)

    logging.info("Tokenizing the dataset")
    ds_tok = ds.map(
        lambda batch: tokenize(batch, tokenizer, key=args.key),
        batched=True,
        num_proc=args.num_proc,
    )

    logging.info("Computing total number of tokens")
    n_tok = sum(ds_tok["n_toks"])
    logging.info(f"Total number of tokens: {n_tok}")

    if args.save_path:
        logging.info("Saving tokenized dataset")
        ds_tok.save_to_disk(args.save_path)
