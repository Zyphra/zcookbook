import jax
import jax.numpy as jnp
import sys
import os
import time

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

from communication.utils import *
from communication.constants import *

def timed_pt2pt(input, args):
    def send_recv(x):
        axis_name = 'i'
        device_id = jax.lax.axis_index(axis_name)
        
        # Step 1: GPU 0 sends to GPU 1
        step1 = jax.lax.ppermute(x, axis_name, [(0, 1)])
        
        # Step 2: GPU 1 sends to GPU 0
        step2 = jax.lax.ppermute(x, axis_name, [(1, 0)])
        
        # Combine results
        result = jax.lax.cond(
            device_id == 0,
            lambda _: step2,  # GPU 0 receives in step 2
            lambda _: step1,  # GPU 1 receives in step 1
            operand=None
        )
        return result

    send_recv_op = jax.pmap(send_recv, axis_name='i')

    # Ensure we're using exactly 2 GPUs
    if jax.device_count() != 2:
        raise ValueError("This benchmark requires exactly 2 GPUs")

    # Warmups
    for _ in range(args.warmups):
        result = send_recv_op(input)
        result.block_until_ready()

    # Time the actual comm op
    start_time = time.time()
    for _ in range(args.trials):
        send_recv_op(input)
        result.block_until_ready()
    end_time = time.time()

    duration = end_time - start_time

    # Maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.nbytes  # Only considering the data sent in one direction
    tput, busbw = get_bw('pt2pt', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.shape[1]}x{input.dtype.itemsize}'

    if not args.raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")

def run_pt2pt(args):
    # Prepare benchmark header
    print_header(args, 'pt2pt')

    if args.scan:
        M_LIST = [2**p for p in range(1, args.maxsize)]

        # Loop over various tensor sizes
        for M in M_LIST:
            try:
                mat = jnp.ones((2, M), dtype=getattr(jnp, args.dtype))
                input = jax.pmap(lambda i, x: x * i)(jnp.arange(2), mat)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print_rank_0('WARNING: Ran out of GPU memory. Exiting comm op.')
                    break
                else:
                    raise e
            timed_pt2pt(input, args)
    else:
        # Send the biggest message size our GPUs can fit
        elements_per_gpu = max_numel(comm_op='pt2pt',
                                     dtype=getattr(jnp, args.dtype),
                                     mem_factor=args.mem_factor * 2,
                                     args=args)
        try:
            mat = jnp.ones((2, elements_per_gpu), dtype=getattr(jnp, args.dtype))
            input = jax.pmap(lambda i, x: x * i)(jnp.arange(2), mat)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print_rank_0('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                return
            else:
                raise e
        timed_pt2pt(input, args)

if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    run_pt2pt(args)

