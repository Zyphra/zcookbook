import sys, os

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

from communication.utils import *
from communication.all_reduce import run_all_reduce
from communication.pt2pt import run_pt2pt
from communication.broadcast import run_broadcast
from communication.constants import *


# For importing
def main(args, rank):

    init_processes(local_rank=rank, args=args)

    ops_to_run = []
    if args.all_reduce:
        ops_to_run.append('all_reduce')
    if args.broadcast:
        ops_to_run.append('broadcast')
    if args.pt2pt:
        ops_to_run.append('pt2pt')

    if len(ops_to_run) == 0:
        ops_to_run = ['all_reduce', 'broadcast', 'pt2pt']

    for comm_op in ops_to_run:
        if comm_op == 'all_reduce':
            run_all_reduce(local_rank=rank, args=args)
        if comm_op == 'pt2pt':
            run_pt2pt(local_rank=rank, args=args)
        if comm_op == 'broadcast':
            run_broadcast(local_rank=rank, args=args)


# For directly calling benchmark
if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    main(args, rank)
