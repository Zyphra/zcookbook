import jax
import jax.numpy as jnp
import sys, os, time
from jax.lib import xla_bridge

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

from communication.utils import *
from communication.constants import *

# Run all_gather and print metrics
def timed_all_gather(input, args):
    def all_gather_op(x):
        return jax.lax.all_gather(x, axis=0, tiled=True)

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        output = jax.jit(all_gather_op)(input)
    sync_all()

    # time the actual comm op trials times and average it
    start_time = time.time()
    for i in range(args.trials):
        output = jax.jit(all_gather_op)(input)
    jax.block_until_ready(output)
    end_time = time.time()
    duration = end_time - start_time

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.dtype.itemsize * input.size
    tput, busbw = get_bw('all_gather', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.size}x{input.dtype.itemsize}'

    if not args.raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")

    return output

def run_all_gather(args):
    # Prepare benchmark header
    print_header(args, 'all_gather')
    global_rank = jax.process_index()
    world_size = jax.device_count()

    if args.scan:
        # Create list of message sizes
        M_LIST = [2**p for p in range(1, args.maxsize)]

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            try:
                mat = jnp.ones((world_size, M), dtype=getattr(jnp, args.dtype))
                sync_all()
                input = (mat * float(global_rank)).reshape(-1)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if global_rank == 0:
                        print('WARNING: Ran out of GPU memory. Exiting comm op.')
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            output = timed_all_gather(input, args)
    else:
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        sync_all()
        elements_per_gpu = max_numel(comm_op='all_gather',
                                     dtype=getattr(jnp, args.dtype),
                                     mem_factor=args.mem_factor,
                                     args=args)
        try:
            mat = jnp.ones(elements_per_gpu, dtype=getattr(jnp, args.dtype))
            # multiply each GPU's tensor by the rank to ease debugging
            input = (mat * float(global_rank)).reshape(-1)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if global_rank == 0:
                    print('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
            else:
                raise e

        sync_all()
        output = timed_all_gather(input, args)

if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    jax.config.update('jax_platform_name', 'gpu')  # Use GPU
    run_all_gather(args)
