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
                            help='conv1d kernel size')
    parser.add_argument("--mamba-ngroups", "-dc",
                        type=int,
                        default=1,
                        help='Number of Mamba groups')
    parser.add_argument("--mamba-headdim", "-dc",
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
                        default=None,
                        help='Number of tokens you are training over')
    parser.add_argument("--no-checkpoint-activations", "-ca",
                        action='store_false',
                        help='Whether activation checkpointing is being used',
                        dest='checkpoint_activations')
    parser.add_argument("--mamba_moe_layers", type = str, default = "")
    return parser

def compute_mamba2_flops(args):
    d_inner = args.hidden_size * args.expand
    Nheads = d_inner // args.mamba_headdim
    mamba2_block_flops = 2 * (2 * d_inner + 2 * args.mamba_ngroups  * args.state_size + Nheads) * args.hidden_size * args.batch_size * args.sequence_length # in proj
    mamba2_block_flops += 2 * args.batch_size * args.sequence_length * (d_inner + 2 * args.mamba_ngroups * args.state_size) * args.conv_dimension * args.d_inner# conv
    mamba2_block_flops += 2 * args.batch_size * args.sequence_length * d_inner * args.state_size * args.d_inner # dtbx
    mamba2_block_flops += 2 * args.batch_size * args.sequence_length * d_inner * args.state_size # ssm state rollover
    mamba2_block_flops += 2 * args.batch_size * args.sequence_length * d_inner * args.state_size * args.d_model # c-> y 
    mamba2_block_flops += args.batch_size * args.sequence_length * args.hidden_size # z gate output
    return mamba2_block_flops

def compute_mamba_flops(args):
    d_inner = args.hidden_size * args.expansion_factor
    dt_rank = math.ceil(args.hidden_size / 16) if args.dt_rank == "auto" else args.dt_rank
    ssm_flops = iter_factor * d_inner * args.tokens * (11 * args.state_size + 4 * dt_rank + 1)
    mamba_projectors_flops = iter_factor * args.tokens * 6 * d_inner * args.hidden_size 
    mamba_conv_flops = iter_factor * args.tokens * 2 * d_inner * args.conv_dimension
    mamba_flops = ssm_flops + mamba_projectors_flops + mamba_conv_flops
    return mamba_flops

def compute_attention_flops(args):
    qkv_flops = int(iter_factor * 2 * (1 + 2 * args.kv_size_ratio) * args.batch_size * args.hidden_size * args.hidden_size)
    attention_matrix_flops = iter_factor * 2 * args.batch_size * args.sequence_length * args.hidden_size
    attention_over_values_flops = iter_factor * 2 * args.batch_size * args.sequence_length * args.hidden_size
    linear_projection_flops = iter_factor * 2 * args.batch_size * args.hidden_size * args.hidden_size
    return qkv_flops + attention_matrix_flops + attention_over_values_flops + linear_projection_flops 

def compute_ffn_flops(args):
    if args.ffn_hidden_size is not None:
        ffn_flops = int(2 * args.batch_size * args.ffn_hidden_size * args.ffn_hidden_size)
    else:
        ffn_flops = int(2  * args.ffn_expansion_factor) * args.batch_size * args.hidden_size * args.hidden_size
    return ffn_flops



# calculates the flops of a model given its hparams
def calc_flops(args):
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

    mamba_flops = compute_mamba_flops(args)
    mamba2_flops = compute_mamba2_flops(args)
    attention_flops = compute_attention_flops(args)
    ffn_flops = compute_ffn_flops(args)

    total_mamba_flops = 0
    total_mamba2_flops = 0
    total_attention_flops = 0
    total_ffn_flops = 0


    # no activation checkpointing for embeddings
    embedding_flops = 6 * args.tokens * args.hidden_size * args.vocab_size

    if args.mamba_moe_layers == "":
        # assume a pure mamba1 model unless specified otherwise
        total_flops = embedding_flops + (mamba_flops * args.num_mamba_layers)
        total_mamba_flops += mamba_flops * args.num_mamba_layers)
        # if MoE layers add these in
        if args.num_moe_layers > 0:
            ffn_flops = iter_factor * args.tokens  * 4 * args.ffn_hidden_size * args.num_moe_layers * args.hidden_size
            if args.swiglu:
                ffn_flops = 3/2 * ffn_flops
            gating_flops = iter_factor * 2 * args.tokens * args.num_experts * args.num_moe_layers
            total_flops += ffn_flops + gating_flops
            total_ffn_flops += ffn_flops
        
    else:
        arch_list = args.mamba_moe_layers.split(" ")
        total_flops = 0
        total_flops += embedding_params
        for el in arch_list:
            if el == "r":
                # mamba layer
                total_flops += mamba_flops
                total_mamba_flops += mamba_flops
            elif el == "m":
                total_flops += mamba2_flops
                total_mamba2_flops += mamba2_flops
            elif el == "a":
                total_flops += attention_flops 
                total_attention_flops += attention_flops
            elif el.isnumeric():
                total_flops += ffn_flops
                total_ffn_flops += ffn_flops
            elif el == "g":
                # zamba shared layer
                original_hidden_size = args.hidden_size
                args.hidden_size = original_hidden_size * 2
                shared_attention_flops = compute_attention_flops(args)
                shared_ffn_flops = compute_ffn_flops(args)
                total_flops += shared_attention_flops
                total_attention_flops += shared_attention_flops
                total_flops += shared_ffn_flops
                total_ffn_flops = shared_ffn_flops
                args.hidden_size = original_hidden_size
                # final downprojector matrix
                total_flops += 4 * args.batch_size * args.sequence_length * args.hidden_size * args.hidden_size
            else:
                raise ValueError("Invalid layer string: " + str(el) " not recognized.")
    
    total_flops *= iter_factor

    print(f'Calculating number of FLOPs with training configuration: {vars(args)}\n')
    print(f'Total Mamba FLOPs: {convert_flops(total_mamba_flops)}')
    print(f'Total Mamba2 FLOPs: {convert_flops(total_mamba2_flops)}')
    print(f'Total Attention FLOPs: {convert_flops(total_attention_flops)}')
    print(f'Total FFN FLOPs: {convert_flops(total_ffn_flops)}')
    print(f'Embedding FLOPs: {convert_flops(embedding_flops)}')
    print(f'Total FLOPs for the Model: {convert_flops(total_flops)}')
    if args.tokens is not None:
        total_flops_through_training = int(total_flops * (args.tokens // args.batch_size))
        print(f'Total FLOPs through training: {convert_flops(total_flops_through_training)}')

if __name__ == "__main__":
    print('\nExample: python calc_mamba_moe_flops.py -num-mamba-layers 12 -hs 768 --num-experts 8 --num-moe-layers 12 -s 2048 --tokens 300e9')
    
    args = config_parser().parse_args()
    calc_flops(args)
