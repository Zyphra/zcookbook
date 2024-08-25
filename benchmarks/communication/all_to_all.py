import jax
import jax.numpy as jnp
import sys, os, time
from jax.lib import xla_bridge

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

from communication.utils import *
from communication.constants import *

def timed_all_to_all(input, output, args):
    def all_to_all_op(x):
        return jax.lax.all_to_all(x, axis=0, split_axis=0, concat_axis=0, num_devices=jax.device_count())

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        output = jax.jit(all_to_all_op)(input)
    sync_all()

    # time the actual comm op trials times and average it
    start_time = time.time()
    for i in range(args.trials):
        output = jax.jit(all_to_all_op)(input)
    jax.block_until_ready(output)
    end_time = time.time()
    duration = end_time - start_time

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.dtype.itemsize * input.size
    n = jax.device_count()
    tput, busbw = get_bw('all_to_all', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.size}x{input.dtype.itemsize}'

    if not args.raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")

    return output

def run_all_to_all(args):
    world_size = jax.device_count()
    global_rank = jax.process_index()
    # Prepare benchmark header
    print_header(args, 'all_to_all')

    if args.scan:
        M_LIST = [2**p for p in range(1, args.maxsize)]

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            try:
                mat = jnp.ones((world_size, M), dtype=getattr(jnp, args.dtype))
                assert mat.size % world_size == 0, f"tensor cannot be divided in {world_size} chunks"
                sync_all()
                input = (mat * float(global_rank)).reshape(-1)
                output = jnp.zeros_like(input)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if global_rank == 0:
                        print('WARNING: Ran out of GPU memory. Exiting comm op.')
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            output = timed_all_to_all(input, output, args)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        elements_per_gpu = max_numel(comm_op='all_to_all',
                                     dtype=getattr(jnp, args.dtype),
                                     mem_factor=args.mem_factor,
                                     args=args)
        try:
            mat = jnp.ones(elements_per_gpu, dtype=getattr(jnp, args.dtype))
            assert mat.size % world_size == 0, f"tensor with {mat.size} elements cannot be divided in {world_size} chunks"
            input = (mat * float(global_rank)).reshape(-1)
            output = jnp.zeros_like(input)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if global_rank == 0:
                    print('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
            else:
                raise e
        sync_all()

        if args.debug:
            for i in range(world_size):
                if i == global_rank:
                    print(f"Before AllToAll Input List at rank {global_rank}: {input}")
                sync_all()

        output = timed_all_to_all(input, output, args)

        if args.debug:
            for i in range(world_size):
                if i == global_rank:
                    print(f"AllToAll Results at rank {global_rank}: {output}")
                sync_all()

if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    jax.config.update('jax_platform_name', 'gpu')  # Use GPU
    run_all_to_all(args)
