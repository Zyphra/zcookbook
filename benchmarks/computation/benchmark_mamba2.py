import argparse
import torch
import math
from mamba_ssm import Mamba2
from utils import time_fwd_bwd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark Mamba2')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32', 'bf16'], help='Data type for torch operations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--seqlen', type=int, default=64, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--d_conv', type=int, default=4, help='Dimension of convolution kernel')
    parser.add_argument('--d_model', type=int, default=1024, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=64, help='State dimension')
    parser.add_argument('--expand', type=int, default=2, help='Expansion factor')
    parser.add_argument('--headdim', type=int, default=128, help='Head dimension')
    parser.add_argument('--ngroups', type=int, default=1, help='Number of mamba2 groups')
    parser.add_argument('--repeats', type=int, default=30, help='Number of repeats for benchmarking')
    parser.add_argument('--device', type=str, default='cuda', help='Torch device to run the benchmark on')
    return parser.parse_args()

def flops_mamba2(hidden_size, expansion_factor, headdim, ngroups, d_state, sequence_length, batch_size, conv_dimension, num_layers, dt_rank="auto", mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    iter_factor = 1 if mode == "fwd" else (2 if mode == "bwd" else 3)
    d_inner = hidden_size * expansion_factor
    Nheads = d_inner // headdim
    mamba2_block_flops = 2 * (2 * d_inner + 2 * ngroups  * d_state + Nheads) * hidden_size * batch_size * sequence_length # in proj
    mamba2_block_flops += 2 * batch_size * sequence_length * (d_inner + 2 * ngroups * d_state) * conv_dimension * d_inner# conv
    mamba2_block_flops += 2 * batch_size * sequence_length * d_inner * d_state * d_inner # dtbx
    mamba2_block_flops += 2 * batch_size * sequence_length * d_inner * d_state # ssm state rollover
    mamba2_block_flops += 2 * batch_size * sequence_length * d_inner * d_state * hidden_size # c-> y
    mamba2_block_flops += batch_size * sequence_length * hidden_size # z gate output
    return mamba2_block_flops

def pretty_print_latency(latency_ms):
    if latency_ms < 1:
        return f"{latency_ms*1000:.2f} µs"
    elif latency_ms < 1000:
        return f"{latency_ms:.2f} ms"
    else:
        return f"{latency_ms/1000:.2f} s"

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def main():
    args = parse_arguments()

    dtype_map = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}
    dtype = dtype_map[args.dtype]

    batch_size = args.batch_size
    seqlen = args.seqlen
    d_model = args.d_model
    d_state = args.d_state
    expand = args.expand
    headdim = args.headdim
    ngroups = args.ngroups
    d_conv = args.d_conv


    input = torch.randn(batch_size, seqlen, d_model, device=args.device, dtype=dtype, requires_grad=True)
    model = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim, device=args.device, dtype=dtype)

    num_params = sum(p.numel() for p in model.parameters())

    # Warmup
    for _ in range(10):
        output = model(input)
        output.sum().backward()

    # Benchmark
    f, b = time_fwd_bwd(model, input, repeats=args.repeats, verbose=args.verbose)


    fwd_flops = flops_mamba2(d_model, expand, headdim, ngroups, d_state, seqlen, batch_size, d_conv, 1, mode="fwd")
    bwd_flops = flops_mamba2(d_model, expand, headdim, ngroups, d_state, seqlen, batch_size, d_conv, 1, mode="bwd")
    fwd_bwd_flops = flops_mamba2(d_model, expand, headdim, ngroups, d_state, seqlen, batch_size, d_conv, 1, mode="fwd_bwd")


    print(f"### d_model={d_model}, d_state={d_state}, expand={expand}, batch_size={batch_size}, seqlen={seqlen} ###")
    print(f"Mamba2 FWD Latency: {pretty_print_latency(f)}")
    print(f"Mamba2 BWD Latency: {pretty_print_latency(b)}")
    print(f"Mamba2 FWD+BWD Latency: {pretty_print_latency(f + b)}")
    print(f"Mamba2 FWD Throughput: {efficiency(fwd_flops, f):.2f} TFLOPs/s")
    print(f"Mamba2 BWD Throughput: {efficiency(bwd_flops, b):.2f} TFLOPs/s")
    print(f"Mamba2 FWD+BWD Throughput: {efficiency(fwd_bwd_flops, f + b):.2f} TFLOPs/s")

if __name__ == "__main__":
    main()
