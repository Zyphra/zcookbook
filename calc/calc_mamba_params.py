import argparse
import math

# Helper function to pretty-print message sizes
def convert_params(params):
    if params == 0:
        return "0"
    size_name = ("", "K", "M", "B", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    return "%s %s" % (s, size_name[i])

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", "-v",
                        type=int,
                        default=50277,
                        help='Size of the vocab')
    parser.add_argument("--d-model", "-dm",
                        type=int,
                        default=768,
                        help='Embedding dimension')
    parser.add_argument("--d-state", "-ds",
                        type=int,
                        default=16,
                        help='Hidded state dimension')
    parser.add_argument("--d-conv", "-dc",
                        type=int,
                        default=4,
                        help='conv1d kernel size')
    parser.add_argument("--mamba-ngroups", "-dc",
                        type=int,
                        default=1,
                        help='Number of Mamba groups')
    parser.add_argument("--mamba-headdim", "-dc",
                        type=int,
                        default=64,
                        help='Mamba2 head dimension')
    parser.add_argument("--expand", "-ex",
                        type=int,
                        default=2,
                        help='Inner state expansion factor')
    parser.add_argument("--dt-rank", "-dr",
                        type=int,
                        default=-1,
                        help='Rank of the delta. Default is -1, which means auto')
    parser.add_argument("--num-layers", "-l",
                        type=int,
                        default=24,
                        help='Total number of sequential layers used in model (both Mamba and MoE')
    parser.add_argument("--num-experts", "-e",
                        type=int,
                        default=0,
                        help="Number of experts used in model")
    parser.add_argument("--ffn-expansion-factor", '-fe',
                        type=float,
                        default=4,
                        help="Expansion factor of the ffn hidden dimension")
    parser.add_argument("--expert-interval", "-ei",
                        type=int,
                        default=1,
                        help="Every N blocks to put an MoE block")
    parser.add_argument("--parallel-moe", "-p",
                        type = bool,
                        default = False,
                        help = "Run the MoE MLPs in parallel with the mamba blocks like gptj")
    parser.add_argument("--swiglu",
                        action="store_true",
                        help='Use swiglu MLP. If set, ffn-hidden-size needs to be specified and is defined as the inner dimension of each of the three MLP weights.')
    parser.add_argument("--ffn-hidden-size",
                        type=int,
                        default=0,
                        help="Hidden dimension of the MLP")
    parser.add_argument("--mamba_moe_layers", type = str, default = "")
    parser.add_argument("--use-global-mem", type=bool, default = False)
    parser.add_argument("--global-memory-projection-interval", type=int, default = 2)
    
    return parser

def compute_mamba_block_params(args):
    d_inner = args.d_model * args.expand
    dt_rank = math.ceil(args.d_model / 16) if args.dt_rank < 1 else args.dt_rank
    
    
    mamba_block_params = (d_inner * args.d_model)              # W_x
    mamba_block_params += (d_inner * args.d_model)             # W_z
    mamba_block_params += (args.d_conv * d_inner) + d_inner    # conv1d
    mamba_block_params += (args.d_state * d_inner)             # W_B
    mamba_block_params += (args.d_state * d_inner)             # W_C
    mamba_block_params += 2 * (dt_rank * d_inner) + d_inner    # W_dt
    mamba_block_params += (d_inner * args.d_state)             # W_A
    mamba_block_params += (d_inner)                            # D
    mamba_block_params += (args.d_model * d_inner)             # W_y
    mamba_block_params += 2 * args.d_model                     # LayerNorm
    return mamba_block_params, dt_rank
    
def compute_mamba2_block_params(args):
    d_inner = args.d_model * args.expand
    n_heads = d_inner / args.mamba_headdim
    d_in_proj = (2 * d_inner) + (2 * args.mamba_ngroups * args.d_state) + n_heads
    mamba2_block_params = args.d_model * d_in_proj # W_in
    mamba2_block_params += 3 * n_heads # A, dt, D
    mamba2_block_params += (d_inner + (2 * args.mamba_ngroups * args.d_state)) * args.d_conv # conv weight
    mamba2_block_params += d_inner + (2 * args.mamba_ngroups * args.d_state) # conv bias
    mamba2_block_params += d_inner # layernorm
    mamba2_block_params += d_inner * args.d_model # W_out
    return mamba2_block_params

# calculates the params of a model given their hparams
def calc_params(args):
    if args.swiglu:
        assert args.ffn_hidden_size > 0, "If args.swiglu=True, ffn-hidden-size needs to be specified."
    # Embedding unembedding weights are tied
    embedding_params = args.d_model * args.vocab_size
    attention_block_params =  4 * args.d_model * args.d_model
    
    mamba_block_params, dt_rank = compute_mamba_block_params(args)
    mamba2_block_params = compute_mamba2_block_params(args)
    

    
    ffn_dim = args.d_model * args.ffn_expansion_factor
    ffn_block_params = 2 * ffn_dim * args.d_model
    
    if args.mamba_moe_layers == "":
        if args.num_experts == 0:
            # pure mamba
            total_ffn_params = 0
            ffn_block_params = 0
            total_expert_params = 0
            total_params = args.num_layers * mamba_block_params + embedding_params
            mamba_block_params = total_params

        else:
            if not args.parallel_moe:
                mamba_block_params = int(round((args.num_layers * mamba_block_params) * (1 - (1/args.expert_interval))))
            else:
                
                mamba_block_params = int(round((args.num_layers * mamba_block_params)))

            if args.swiglu:
                ffn_block_params = 3 * args.ffn_hidden_size * args.d_model
            if not args.parallel_moe:
                total_ffn_params = (args.num_layers // args.expert_interval) * ffn_block_params
            else:
                total_ffn_params = args.num_layers  * ffn_block_params
            total_expert_params = total_ffn_params * args.num_experts
            total_params = mamba_block_params + total_expert_params + embedding_params
            forward_pass_params = mamba_block_params + total_ffn_params + embedding_params
    else:
        arch_list = args.mamba_moe_layers.split(" ")
        len_list = len(arch_list)
        assert len_list == args.num_layers, "Length of mamba moe list is not the same as the total number of layers"
        total_params = 0
        total_params += embedding_params
        forward_pass_params = 0
        forward_pass_params += embedding_params
        total_attention_params = 0
        total_mamba_params = 0
        total_ffn_params = 0
        for el in arch_list:
            if el == 'r':
                total_params += mamba_block_params
                forward_pass_params += mamba_block_params
                total_mamba_params += mamba_block_params
            elif el == 'm':
                total_params += mamba2_block_params
                forward_pass_params += mamba2_block_params
                total_mamba_params += mamba2_block_params
            elif el == 'a':
                total_params += attention_block_params
                forward_pass_params += attention_block_params
                total_attention_params += attention_block_params
            elif el.isnumeric():
                num_experts = int(el)
                total_params  += num_experts * ffn_block_params
                forward_pass_params += ffn_block_params
                total_ffn_params += num_experts * ffn_block_params
            else:
                raise ValueError("Invalid layers string")
            
    if args.use_global_mem:
        # we add a transformer layer and an MLP layer to the model plus we add a d^2 linear layer to each layer
        global_mem_params = 0
        global_mem_params += 3 * (2 * args.d_model)**2 + 2 * args.d_model**2 # qkv act on 2d_model, output proj maps 2d_model -> d_model
        global_mem_params += 12 * args.d_model # the 12 is because we use swiglu without the 2/3 resizing of the ffn hidden dimension
        global_mem_projection_params = (args.num_layers // args.global_memory_projection_interval) * args.d_model * args.d_model
        total_params += global_mem_params 
        total_params += global_mem_projection_params
        
            

    if args.mamba_moe_layers == "":
        print(f'Calculating number of parameters with training configuration: {vars(args)}\n')
        print(f'dt_rank: {convert_params(dt_rank)}')
        print(f'Embedding parameters: {convert_params(embedding_params)}')
        print(f'Single Mamba block params: {convert_params(mamba_block_params)}')
        print(f'FFN block params: {convert_params(ffn_block_params)}')
        print(f'Total Mamba Params: {convert_params(mamba_block_params)}')
        print(f"Total FFN params: {convert_params(total_expert_params)}")

    else:
        print(f'Calculating number of parameters with training configuration: {vars(args)}\n')
        print(f'Embedding parameters: {convert_params(embedding_params)}')
        print(f'Total Mamba Params: {convert_params(total_mamba_params)}')
        print(f'Total Attention Params: {convert_params(total_attention_params)}')
        print(f"Total FFN params: {convert_params(total_ffn_params)}")
        
    if args.use_global_mem:
        print(f'Global Memory Parameters: {convert_params(global_mem_params)}')
        print(f'Global Memory Projection Params: {convert_params(global_mem_projection_params)}')
        
    print(f'Total params: {convert_params(total_params)}')
    print("Aspect Ratio: ", args.d_model / args.num_layers)
    if args.num_experts > 0:
        print(f'Forward pass params: {convert_params(forward_pass_params)}')
        
    

if __name__ == "__main__":
    args = config_parser().parse_args()
    calc_params(args)
