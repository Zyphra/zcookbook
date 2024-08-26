# Computation Benchmarks

This directory contains isolated benchmarks for the core blocks of Zamba hybrid models: attention and Mamba. These benchmarks are designed to compare accelerators, find [good model sizes](https://arxiv.org/abs/2401.14489), and test optimizations.

## Available Benchmarks

1. Mamba1 Benchmark (`benchmark_mamba.py`)
2. Flash Attention Benchmark (`benchmark_flash_attention.py`)
3. Mamba2 Benchmark (`benchmark_mamba2.py`)

## Running Benchmarks

To run a benchmark, use the following command:

```
python <benchmark_file>.py
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
