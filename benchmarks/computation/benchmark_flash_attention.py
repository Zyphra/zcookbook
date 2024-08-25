# Benchmark the Flash Attention block
# Based on https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py

import argparse
import torch
import math
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark Flash Attention')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32', 'fp64', 'bf16'], help='Data type for torch operations')
    parser.add_argument('--causal', type=str, nargs='+', choices=['true', 'false'], default=['false'], help='Enable causal masking')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--dropout_p', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--seqlen', type=int, default=1024, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--nheads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Size of each attention head')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Total hidden dimension')
    parser.add_argument('--repeats', type=int, default=30, help='Number of repeats for benchmarking')
    parser.add_argument('--device', type=str, default='cuda', help='Torch device to run the benchmark on')
    return parser.parse_args()

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def pretty_print_latency(latency_ms):
    if latency_ms < 1:
        return f"{latency_ms*1000:.2f} µs"
    elif latency_ms < 1000:
        return f"{latency_ms:.2f} ms"
    else:
        return f"{latency_ms/1000:.2f} s"

def main():
    args = parse_arguments()

    dtype_map = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64, 'bf16': torch.bfloat16}
    dtype = dtype_map[args.dtype]
    causal = args.causal[0] == 'true'

    batch_size = args.batch_size
    seqlen = args.seqlen
    nheads = args.nheads
    head_dim = args.head_dim
    hidden_dim = args.hidden_dim

    # Ensure hidden_dim is divisible by nheads
    assert hidden_dim % nheads == 0, f"hidden_dim {hidden_dim} must be divisible by nheads {nheads}"

    qkv = torch.randn(batch_size, seqlen, 3, nheads, head_dim, device=args.device, dtype=dtype, requires_grad=True)

    # Warmup
    for _ in range(10):
        output = flash_attn_qkvpacked_func(qkv, args.dropout_p, causal=causal)
        output.sum().backward()

    # Benchmark forward pass
    fwd_times = benchmark_forward(
        flash_attn_qkvpacked_func, qkv, args.dropout_p, causal=causal,
        repeats=args.repeats
    )

    # Benchmark backward pass
    bwd_times = benchmark_backward(
        flash_attn_qkvpacked_func, qkv, args.dropout_p, causal=causal,
        repeats=args.repeats
    )

    f, b = fwd_times[1], bwd_times[1]

    fwd_flops = flops(batch_size, seqlen, head_dim, nheads, causal, mode="fwd")
    bwd_flops = flops(batch_size, seqlen, head_dim, nheads, causal, mode="bwd")
    fwd_bwd_flops = flops(batch_size, seqlen, head_dim, nheads, causal, mode="fwd_bwd")

    print(f"### Causal={causal}, head_dim={head_dim}, batch_size={batch_size}, seqlen={seqlen}, hidden_dim={hidden_dim} ###")
    print(f"FlashAttention2 FWD Latency: {pretty_print_latency(f)}")
    print(f"FlashAttention2 BWD Latency: {pretty_print_latency(b)}")
    print(f"FlashAttention2 FWD+BWD Latency: {pretty_print_latency(f + b)}")
    print(f"FlashAttention2 FWD Throughput: {efficiency(fwd_flops, f):.2f} TFLOPs/s")
    print(f"FlashAttention2 BWD Throughput: {efficiency(bwd_flops, b):.2f} TFLOPs/s")
    print(f"FlashAttention2 FWD+BWD Throughput: {efficiency(fwd_bwd_flops, f + b):.2f} TFLOPs/s")

if __name__ == "__main__":
    main()
