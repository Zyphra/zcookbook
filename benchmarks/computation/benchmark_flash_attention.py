# Benchmark the Flash Attention block
# Based on https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py

import argparse
import numpy as np
import torch
from utils import time_fwd_bwd, convert_params
from flash_attn import flash_attn_qkvpacked_func

parser = argparse.ArgumentParser(description='Benchmark Flash Attention')
parser.add_argument('--dtype', type=str, default='fp16', help='Data type for torch operations (fp16, fp32, fp64, bf16)')
parser.add_argument('--causal', type=str, nargs='+', choices=['true', 'false'], default=['true', 'false'], help='Enable causal masking (true or false)')
parser.add_argument('--verbose', '-v', type=bool, default=False, help='Enable verbose output')
parser.add_argument('--dropout_p', type=float, default=0.0, help='Dropout probability')
parser.add_argument('--seqlen', type=int, nargs='+', default=[4096], help='Sequence lengths to benchmark')
parser.add_argument('--seqlen_range', type=int, nargs=3, default=None, help='Range of sequence lengths to benchmark (start, stop, step)')
parser.add_argument('--batch_size', type=int, nargs='+', default=[1], help='Batch sizes to benchmark')
parser.add_argument('--batch_size_range', type=int, nargs=3, default=None, help='Range of batch sizes to benchmark (start, stop, step)')
parser.add_argument('--nheads', type=int, nargs='+', default=[32], help='Number of attention heads')
parser.add_argument('--nheads_range', type=int, nargs=3, default=None, help='Range of number of attention heads to benchmark (start, stop, step)')
parser.add_argument('--head_dim', type=int, nargs='+', default=None, help='Size of each attention head')
parser.add_argument('--head_dim_range', type=int, nargs=3, default=None, help='Range of sizes of each attention head to benchmark (start, stop, step)')
parser.add_argument('--hidden_dim', type=int, nargs='+', default=None, help='Total hidden dimension')
parser.add_argument('--hidden_dim_range', type=int, nargs=3, default=None, help='Range of total hidden dimensions to benchmark (start, stop, step)')
parser.add_argument('--repeats', type=int, default=30, help='Number of repeats for benchmarking')
parser.add_argument('--device', type=str, default='cuda', help='Torch device to run the benchmark on')
args = parser.parse_args()

# Assertions to ensure only one of each range or non-range argument is provided
assert (args.seqlen is not None) ^ (args.seqlen_range is not None), "Provide either seqlen or seqlen_range, not both"
assert (args.batch_size is not None) ^ (args.batch_size_range is not None), "Provide either batch_size or batch_size_range, not both"
assert (args.nheads is not None) ^ (args.nheads_range is not None), "Provide either nheads or nheads_range, not both"
assert (args.head_dim is not None) ^ (args.head_dim_range is not None), "Provide either head_dim or head_dim_range, not both"
assert (args.hidden_dim is not None) ^ (args.hidden_dim_range is not None), "Provide either hidden_dim or hidden_dim_range, not both"

# Populate nheads, batch_sizes, and seqlens based on parsed arguments
nheads = list(range(*args.nheads_range)) if args.nheads_range else args.nheads
batch_sizes = list(range(*args.batch_size_range)) if args.batch_size_range else args.batch_size
seqlens = list(range(*args.seqlen_range)) if args.seqlen_range else args.seqlen
headdims = list(range(*args.head_dim_range)) if args.head_dim_range else [args.head_dim] if args.head_dim is not None else []
hidden_dims = list(range(*args.hidden_dim_range)) if args.hidden_dim_range else [args.hidden_dim] if args.hidden_dim is not None else []

# Ensure hidden_dim is divisible by headdim
for hidden_dim in hidden_dims:
    for headdim in headdims:
        assert hidden_dim % headdim == 0, f"hidden_dim {hidden_dim} must be divisible by headdim {headdim}"

# Convert string argument to torch dtype
dtype_map = {
    'fp16': torch.float16,
    'fp32': torch.float32,
    'fp64': torch.float64,
    'bf16': torch.bfloat16
}
dtype = dtype_map.get(args.dtype, torch.float16)  # Default to float16 if unrecognized
dropout_p = args.dropout_p
# Convert the parsed argument to a list of booleans
causal_vals = [x == 'true' for x in args.causal]

latency_f = {}
latency_b = {}
latency_f_b = {}
throughput_f = {}
throughput_b = {}
throughput_f_b = {}
for causal in causal_vals:
    for headdim in headdims:
        for nhead in nheads:
            for batch_size in batch_sizes:
                for seqlen in seqlens:
                    config = (causal, args.headdim, batch_size, seqlen)
                    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=args.device, dtype=dtype, requires_grad=True)
                    num_params = 4*args.hidden_dim**2
                    try:
                        f, b = time_fwd_bwd(
                            flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=args.repeats, verbose=args.verbose
                        )
                    except Exception as e: # Account for OOM
                        f, b = float('nan'), float('nan')
                        continue
                    latency_f[config] = f
                    latency_b[config] = b

                    print(f"### Causal={causal}, headdim={args.headdim}, batch_size={batch_size}, seqlen={seqlen}, params per block={convert_params(num_params)} ###")

                    latency_f_b[config] = latency_f[config] + latency_b[config]

                    throughput_f[config] = ns_per_param(latency_f[config], num_params)
                    throughput_b[config] = ns_per_param(latency_b[config], num_params)
                    throughput_f_b[config] = ns_per_param(latency_f_b[config], num_params)
                
                    print(
                        f"FlashAttention2 FWD Throughput: {throughput_f[config]:.5f} ns/param, "
                        f"BWD Throughput: {throughput_b[config]:.5f} ns/param, "
                        f"FWD + BWD Throughput: {throughput_f_b[config]:.5f} ns/param"
                    )