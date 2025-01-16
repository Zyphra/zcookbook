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
    
    # Calculate scientific notation
    sci_exp = int(math.floor(math.log10(params)))
    sci_coeff = round(params / (10 ** sci_exp), 2)
    sci_notation = f"{sci_coeff} × 10^{sci_exp}"
    
    return f"{s} {size_name[i]} ({sci_notation} FLOPs)"

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
    parser.add_argument("--batch-size", "-b",
                        type=int,
                        default=32,
                        help='Training batch size')
    parser.add_argument("--kv-size-ratio", "-kv",
                        type=float,
                        default=1.0,
                        help='Ratio of kv heads to query heads used in model. 1.0 for MHA')
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
                        help='Expansion factor relating inner dimension and hidden size')
    parser.add_argument("--conv-dimension",
                        type=int,
                        default=4,
                        help='Dimension of convolution kernel')    
    parser.add_argument("--dt-rank",
                        type=str,
                        default="auto",
                        help='Rank of dt')
    parser.add_argument("--mamba-ngroups",
                        type=int,
                        default=1,
                        help='Number of Mamba groups')
    parser.add_argument("--mamba-headdim",
                        type=int,
                        default=64,
                        help='Mamba2 head dimension')
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
                        help='Hidden dimension of the FFN')
    parser.add_argument("--ffn-expansion-factor", "-ff",
                        type=int,
                        default=4,
                        help='How much the MLP hidden size expands')
    parser.add_argument("--num-mlp-linears", "-nl",
                        type=int,
                        default=2,
                        help='How many linear layers per MLP block. Set to 3 for SwiGLU or GEGLU Llama-style gated MLPs.')
    parser.add_argument("--topk", "-t",
                        type=int,
                        default=1,
                        help='Top k routing for MoE')
    parser.add_argument("--swiglu",
                        action="store_true",
                        help='Use swiglu FFN')    
    parser.add_argument("--tokens",
                        type=float,
                        default=300e9,
                        help='Number of tokens you are training over')
    parser.add_argument("--no-checkpoint-activations", "-ca",
                        action='store_false',
                        help='Whether activation checkpointing is being used',
                        dest='checkpoint_activations')
    parser.add_argument("--mamba-moe-layers",
                        type=str,
                        default="",
                        help='Layer configuration string')
    return parser

def compute_mamba1_flops(args, iter_factor):
    d_inner = args.hidden_size * args.expansion_factor
    dt_rank = math.ceil(args.hidden_size / 16) if args.dt_rank == "auto" else int(args.dt_rank)
    
    # SSM state computation
    ssm_flops = iter_factor * d_inner * (11 * args.state_size + 4 * dt_rank + 1)
    # Input and output projections
    mamba_projectors_flops = iter_factor * 6 * d_inner * args.hidden_size 
    # Convolution operations
    mamba_conv_flops = iter_factor * 2 * d_inner * args.conv_dimension
    
    mamba1_flops = ssm_flops + mamba_projectors_flops + mamba_conv_flops
    return mamba1_flops * args.tokens

def compute_mamba2_flops(args, iter_factor):
    d_inner = args.hidden_size * args.expansion_factor
    Nheads = d_inner // args.mamba_headdim
    
    # Input projections
    mamba2_block_flops = 2 * (2 * d_inner + 2 * args.mamba_ngroups * args.state_size + Nheads) * args.hidden_size
    # Convolution computations
    mamba2_block_flops += 2 * (d_inner + 2 * args.mamba_ngroups * args.state_size) * args.conv_dimension + (d_inner + 2 * args.mamba_ngroups * args.state_size)
    # S4D core computations
    mamba2_block_flops += 4* d_inner * args.state_size
    # State updates
    mamba2_block_flops += 2 * d_inner * args.state_size
    # Multiply state by C and add Dx
    mamba2_block_flops += d_inner * (2 + args.state_size)
    # Gated norm (gate activation + gate-state product + rms norm)
    mamba2_block_flops += (d_inner + d_inner + 3 * d_inner)
    # Output projections
    mamba2_block_flops += 2 * d_inner * args.state_size * args.hidden_size
    # Final gating
    mamba2_block_flops += args.hidden_size
    return mamba2_block_flops * args.tokens

def compute_attention_flops(args, iter_factor):
    # An A_(m x k) X B_(k x n) matrix multiplication requires 2m x k x n FLOPs (multiplies and adds)
    qkv_flops = int(iter_factor * 2 * (1 + 2 * args.kv_size_ratio) * args.hidden_size * args.hidden_size)
    attention_matrix_flops = iter_factor * 2 * args.hidden_size
    attention_over_values_flops = iter_factor * 2 * args.hidden_size
    linear_projection_flops = iter_factor * 2 * args.hidden_size * args.hidden_size
    return args.tokens * (qkv_flops + attention_matrix_flops + attention_over_values_flops + linear_projection_flops)

def compute_ffn_flops(args, iter_factor):
    # If custom FFN hidden size is provided, use that
    if args.ffn_hidden_size is not None:
        intermediate_dim = args.ffn_hidden_size
    else:
        # Otherwise use the expansion factor
        intermediate_dim = args.hidden_size * args.ffn_expansion_factor
    
    # Calculate FFN FLOPs based on number of linear layers
    ffn_flops = int(iter_factor * 2 * args.num_mlp_linears * 
                    args.hidden_size * intermediate_dim)
    
    if args.swiglu:
        # For SwiGLU, add 50% more FLOPs to account for the extra gating computation
        ffn_flops = int(ffn_flops * 1.5)
        
    return ffn_flops * args.tokens

def calc_flops(args):
    if args.num_experts > 1:
        assert args.topk <= args.num_experts, "You cannot route to more experts than you have!"
    if args.tokens is not None:
        args.tokens = int(args.tokens)

    # determine the flops factor. 
    # If no activation checkpointing: 1 for fwd and 2 for bwd
    # If activation checkpointing: add 1 more for recomputation
    iter_factor = 4 if args.checkpoint_activations else 3
    
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size

    # Calculate component FLOPs
    mamba1_flops = compute_mamba1_flops(args, iter_factor)
    mamba2_flops = compute_mamba2_flops(args, iter_factor)
    attention_flops = compute_attention_flops(args, iter_factor)
    ffn_flops = compute_ffn_flops(args, iter_factor)

    # Initialize tracking variables
    total_mamba1_flops = 0
    total_mamba2_flops = 0
    total_attention_flops = 0
    total_ffn_flops = 0

    # No activation checkpointing for embeddings
    embedding_flops = 6 * args.tokens * args.hidden_size * args.vocab_size

    if args.mamba_moe_layers == "":
        # assume a pure mamba1 model unless specified otherwise
        total_flops = embedding_flops + (mamba1_flops * args.num_mamba_layers)
        total_mamba1_flops = mamba1_flops * args.num_mamba_layers
        
        # if MoE layers add these in
        if args.num_moe_layers > 0:
            ffn_flops = iter_factor * args.tokens * 4 * args.ffn_hidden_size * args.num_moe_layers * args.hidden_size
            if args.swiglu:
                ffn_flops = 3/2 * ffn_flops
            gating_flops = iter_factor * 2 * args.tokens * args.num_experts * args.num_moe_layers
            total_flops += ffn_flops + gating_flops
            total_ffn_flops += ffn_flops
        
    else:
        total_flops = embedding_flops
        for layer_type in args.mamba_moe_layers.split():
            if layer_type == "r":
                # mamba1 layer
                total_flops += mamba1_flops
                total_mamba1_flops += mamba1_flops
            elif layer_type == "m":
                # mamba2 layer
                total_flops += mamba2_flops
                total_mamba2_flops += mamba2_flops
            elif layer_type == "a":
                # attention layer
                total_flops += attention_flops 
                total_attention_flops += attention_flops
            elif layer_type.isnumeric():
                # FFN layer
                total_flops += ffn_flops
                total_ffn_flops += ffn_flops
            elif layer_type == "g":
                # shared layer case
                original_hidden_size = args.hidden_size
                args.hidden_size = original_hidden_size * 2
                shared_attention_flops = compute_attention_flops(args, iter_factor)
                shared_ffn_flops = compute_ffn_flops(args, iter_factor)
                total_flops += shared_attention_flops + shared_ffn_flops
                total_attention_flops += shared_attention_flops
                total_ffn_flops += shared_ffn_flops
                args.hidden_size = original_hidden_size
                # final downprojector matrix
                total_flops += 4 * args.hidden_size * args.hidden_size
            else:
                raise ValueError(f"Invalid layer type: {layer_type}")
    
    # Print results
    print(f'\nCalculating number of FLOPs with training configuration: {vars(args)}\n')
    print(f'Total Mamba1 FLOPs: {convert_flops(total_mamba1_flops)}')
    print(f'Total Mamba2 FLOPs: {convert_flops(total_mamba2_flops)}')
    print(f'Total Attention FLOPs: {convert_flops(total_attention_flops)}')
    print(f'Total FFN FLOPs: {convert_flops(total_ffn_flops)}')
    print(f'Embedding FLOPs: {convert_flops(embedding_flops)}')
    print(f'Total FLOPs for the Model: {convert_flops(total_flops)}')

if __name__ == "__main__":
    print('\nExample: python calc_mamba_flops.py --num-mamba-layers 12 -hs 768 --num-experts 8 --num-moe-layers 12 -s 2048 --tokens 300e9')
    args = config_parser().parse_args()
    calc_flops(args)