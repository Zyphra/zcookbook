import argparse
import torch
import math
from mamba_ssm import Mamba
from benchmarks.computation.utils import time_fwd_bwd, ns_per_param

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark Mamba')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32', 'bf16'], help='Data type for torch operations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--seqlen', type=int, default=1024, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--d_model', type=int, default=2048, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=64, help='State dimension')
    parser.add_argument('--expand', type=int, default=2, help='Expansion factor')
    parser.add_argument('--repeats', type=int, default=30, help='Number of repeats for benchmarking')
    parser.add_argument('--device', type=str, default='cuda', help='Torch device to run the benchmark on')
    return parser.parse_args()

def flops_mamba(hidden_size, expansion_factor, state_size, seqlen, batch_size, conv_dimension, num_layers, dt_rank="auto", mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    iter_factor = 3 if mode == "fwd" else 4
    d_inner = hidden_size * expansion_factor
    dt_rank = math.ceil(hidden_size / 16) if dt_rank == "auto" else dt_rank
    ssm_flops = iter_factor * d_inner * seqlen * batch_size * (11 * state_size + 4 * dt_rank + 1) * num_layers
    mamba_projectors_flops = iter_factor * seqlen * batch_size * 6 * d_inner * hidden_size * num_layers
    mamba_conv_flops = iter_factor * seqlen * batch_size * 2 * d_inner * conv_dimension * num_layers
    mamba_flops = ssm_flops + mamba_projectors_flops + mamba_conv_flops
    return mamba_flops

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

    dtype_map = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}
    dtype = dtype_map[args.dtype]

    batch_size = args.batch_size
    seqlen = args.seqlen
    d_model = args.d_model
    d_state = args.d_state
    expand = args.expand

    input = torch.randn(batch_size, seqlen, d_model, device=args.device, dtype=dtype, requires_grad=True)
    model = Mamba(d_model=d_model, d_state=d_state, expand=expand, device=args.device, dtype=dtype)

    num_params = sum(p.numel() for p in model.parameters())

    # Warmup
    for _ in range(10):
        output = model(input)
        output.sum().backward()

    # Benchmark
    f, b = time_fwd_bwd(model, input, repeats=args.repeats, verbose=args.verbose)

    fwd_flops = flops_mamba(batch_size, seqlen, d_model, d_state, expand, mode="fwd")
    bwd_flops = flops_mamba(batch_size, seqlen, d_model, d_state, expand, mode="bwd")
    fwd_bwd_flops = flops_mamba(batch_size, seqlen, d_model, d_state, expand, mode="fwd_bwd")

    print(f"### d_model={d_model}, d_state={d_state}, expand={expand}, batch_size={batch_size}, seqlen={seqlen} ###")
    print(f"Mamba FWD Latency: {pretty_print_latency(f)}")
    print(f"Mamba BWD Latency: {pretty_print_latency(b)}")
    print(f"Mamba FWD+BWD Latency: {pretty_print_latency(f + b)}")
    print(f"Mamba FWD Throughput: {efficiency(fwd_flops, f):.2f} TFLOPs/s")
    print(f"Mamba BWD Throughput: {efficiency(bwd_flops, b):.2f} TFLOPs/s")
    print(f"Mamba FWD+BWD Throughput: {efficiency(fwd_bwd_flops, f + b):.2f} TFLOPs/s")
    print(f"Mamba FWD Speed: {ns_per_param(f, num_params):.5f} ns/param")
    print(f"Mamba BWD Speed: {ns_per_param(b, num_params):.5f} ns/param")
    print(f"Mamba FWD+BWD Speed: {ns_per_param(f + b, num_params):.5f} ns/param")

if __name__ == "__main__":
    main()
