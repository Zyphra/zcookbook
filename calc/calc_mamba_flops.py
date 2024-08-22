import argparse
import math

# Helper function to pretty-print message sizes
def convert_flops(params):
    if params == 0:
        return "0"
    size_name = ("", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs", "EFLOPs", "ZFLOPs", "YFLOPs")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    return "%s %s" % (s, size_name[i])

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", "-v",
                        type=int,
                        default=51200,
                        help='Size of the vocab')
    parser.add_argument("--hidden-size", "-hs",
                        type=int,
                        default=768,
                        help='Dimension of the model\'s hidden size')
    parser.add_argument("--sequence-length", "-s",
                        type=int,
                        default=2048,
                        help='Sequence length used for training')
    parser.add_argument("--num-mamba-layers",
                        type=int,
                        default=0,
                        help='Number of mamba layers used in model')
    parser.add_argument("--state-size",
                        type=int,
                        default=16,
                        help='State dimension')
    parser.add_argument("--expansion-factor", 
                        type=int,
                        default=2,
                        help='Expansion factor relating inner dimension and hidden size (or d_model)')
    parser.add_argument("--conv-dimension",
                        type=int,
                        default=4,
                        help='Dimension of convolution kernel')    
    parser.add_argument("--dt-rank",
                        type=str,
                        default="auto",
                        help='Rank of dt')    
    parser.add_argument("--num-moe-layers", "-l",
                        type=int,
                        default=0,
                        help='Number of moe layers used in model')
    parser.add_argument("--num-experts", "-e",
                    type=int,
                    default=0,
                    help='Number of experts for MoE')
    parser.add_argument("--ffn-hidden-size",
                        type=int,
                        default=None,
                        help='Hidden dimension of the MLP')
    parser.add_argument("--topk", "-t",
                        type=int,
                        default=1,
                        help='Top k routing for MoE')
    parser.add_argument("--swiglu",
                    action="store_true",
                    help='Use swiglu MLP. If set, ffn-hidden-size is defined as the inner dimension of each of the three MLP weights.')    
    parser.add_argument("--tokens",
                        type=int,
                        default=300e9,
                        help='Number of tokens you are training over')
    parser.add_argument("--no-checkpoint-activations", "-ca",
                        action='store_false',
                        help='Whether Megatron-style activation checkpointing is being used',
                        dest='checkpoint_activations')
    return parser

# calculates the flops of a model given its hparams
def calc_params(args):
    if args.num_experts > 1:
        assert args.topk <= args.num_experts, "You cannot route to more experts than you have!"
    

    # An A_(m x k) X B_(k x n) matrix multiplication requires 2m x k x n FLOPs (factor of 2 needed to account for multiplies and adds)

    # determine the flops factor. 
    # If no activation checkpointing/recomputation, 1 for fwd and 2 for bwd (because we need to calculate the grads with respect to both the input and weight tensors). 
    # If activation checkpointing/recomputation, add 1 more for the next full forward pass
    iter_factor = 3
    if args.checkpoint_activations:
        iter_factor += 1
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4* args.hidden_size

    d_inner = args.hidden_size * args.expansion_factor
    dt_rank = math.ceil(args.hidden_size / 16) if args.dt_rank == "auto" else args.dt_rank
    ssm_flops = iter_factor * d_inner * args.tokens * (11 * args.state_size + 4 * dt_rank + 1) * args.num_mamba_layers
    mamba_projectors_flops = iter_factor * args.tokens * 6 * d_inner * args.hidden_size * args.num_mamba_layers
    mamba_conv_flops = iter_factor * args.tokens * 2 * d_inner * args.conv_dimension * args.num_mamba_layers
    mamba_flops = ssm_flops + mamba_projectors_flops + mamba_conv_flops
    # no activation checkpointing for embeddings
    embedding_flops = 6 * args.tokens * args.hidden_size * args.vocab_size

    total_flops = embedding_flops + mamba_flops

    if args.num_moe_layers > 0:
        ffn_flops = iter_factor * args.tokens  * 4 * args.ffn_hidden_size * args.num_moe_layers * args.hidden_size
        if args.swiglu:
            ffn_flops = 3/2 * ffn_flops
        gating_flops = iter_factor * 2 * args.tokens * args.num_experts * args.num_moe_layers
        total_flops += ffn_flops + gating_flops

    print(f'Calculating number of FLOPs with training configuration: {vars(args)}\n')
    print(f'SSM FLOPs: {convert_flops(ssm_flops)}')
    print(f'Mamba projectors FLOPs: {convert_flops(mamba_projectors_flops)}')
    print(f'Mamba pre-SSM convolution FLOPs: {convert_flops(mamba_conv_flops)}')
    print(f'Total Mamba FLOPs: {convert_flops(mamba_flops)}')
    if args.num_moe_layers > 0:
        print(f'MoE FFN FLOPs: {convert_flops(ffn_flops)}')
        print(f'MoE gating FLOPs: {convert_flops(gating_flops)}')
    print(f'Embedding FLOPs: {convert_flops(embedding_flops)}')
    print(f'Total FLOPs for the Model: {convert_flops(total_flops)}')

if __name__ == "__main__":
    print('\nExample: python calc_mamba_moe_flops.py -num-mamba-layers 12 -hs 768 --num-experts 8 --num-moe-layers 12 -s 2048 --tokens 300e9')
    
    args = config_parser().parse_args()
    calc_params(args)
