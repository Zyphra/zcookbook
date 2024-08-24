import multiprocessing as mp
import argparse
import datasets
import os
import more_itertools

import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)


def save_shard(data, path, total, indices):
    for idx in indices:
        shard = data.shard(num_shards=total, index=idx, contiguous=True)
        save_path = os.path.join(path, f"partition_{idx}.jsonl")
        shard.to_json(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=str, required=True, help="Path of HF dataset")
    parser.add_argument("--hf-dir", type=str, default=None, help="Dir in HF dataset")
    parser.add_argument("--save-path", type=str, required=True, help="Folder to save partitioned JSONL dataset to")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of processes for HF processing")
    parser.add_argument("--num-partitions", type=int, default=1, help="Number of partitions to split the dataset into")
    args = parser.parse_args()

    logging.info("Loading the dataset")
    ds = datasets.load_dataset(
        path=args.hf_path,
        data_dir=args.hf_dir,
        num_proc=args.num_proc,
        split="train",
        trust_remote_code=True
    )

    logging.info("Saving JSONL partitions")
    n_proc = min(args.num_proc, args.num_partitions)
    inds_distr = more_itertools.distribute(n_proc, range(args.num_partitions))
    processes = []
    for process_inds in inds_distr:
        p = mp.Process(target=save_shard, args=(ds, args.save_path, args.num_partitions, process_inds))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    logging.info("Done!")
