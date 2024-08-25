Here's an updated README to reflect the new JAX-only implementation:

# JAX Communication Benchmarks

The intent of these benchmarks is to measure communication latency/bandwidth of JAX collective communication operations at the Python layer. These benchmarks are complementary to C-level comms benchmarks like [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) and [NCCL Tests](https://github.com/NVIDIA/nccl-tests) in that users can:
- Easily debug which layer of the communication software stack hangs or performance degradations originate from.
- Measure the expected communication performance of JAX collective operations.

To run benchmarks, there are two options:

1. Run a single communication operation:

For example, run with a single large message size (calculated to barely fit within GPU mem):
<pre>
python all_reduce.py
</pre>

Scan across message sizes:
<pre>
python all_reduce.py --scan
</pre>

2. Run all available communication benchmarks:

<pre>
python run_all.py
</pre>

Like the individual benchmarks, `run_all.py` supports scanning arguments for the max message size, bandwidth-unit, etc. Simply pass the desired arguments to `run_all.py` and they'll be propagated to each comm op.

Finally, users can choose specific communication operations to run in `run_all.py` by passing them as arguments (all operations are run by default). For example:

<pre>
python run_all.py --scan --all-reduce --broadcast
</pre>

There is a wide range of arguments available:

```
usage: run_all.py [-h] [--trials TRIALS] [--warmups WARMUPS] [--maxsize MAXSIZE]
                  [--bw-unit {Gbps,GBps}] [--scan] [--raw] [--all-reduce] [--broadcast] [--dtype DTYPE] [--mem-factor MEM_FACTOR] [--debug]

options:
  -h, --help            show this help message and exit
  --trials TRIALS       Number of timed iterations
  --warmups WARMUPS     Number of warmup (non-timed) iterations
  --maxsize MAXSIZE     Max message size as a power of 2
  --bw-unit {Gbps,GBps}
  --scan                Enables scanning all message sizes
  --raw                 Print the message size and latency without units
  --all-reduce          Run all_reduce
  --broadcast           Run broadcast
  --dtype DTYPE         JAX array dtype
  --mem-factor MEM_FACTOR
                        Proportion of max available GPU memory to use for single-size evals
  --debug               Enables all_to_all debug prints
```

# Adding Communication Benchmarks

To add new communication benchmarks, follow this general procedure:

1. Copy a similar benchmark file (e.g. to add `reduce_scatter`, copy `all_reduce.py` as a template)
2. Add a new bandwidth formula in `utils.get_bandwidth`, a new maximum array element formula in `utils.max_numel`, and a new arg in `utils.benchmark_parser`
3. Replace comm op calls in new file with find-replace
4. Find a good default `mem_factor` for use in `run_<collective>()` function
5. Add new comm op to `run_all.py`

Note: This JAX implementation doesn't require MPI or a specific launcher. It uses JAX's built-in multi-device support. Make sure you have JAX with GPU support installed if you're running on GPUs.