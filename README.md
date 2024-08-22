# The Zyphra Training Cookbook
By Quentin Anthony, Beren Millidge, Paolo Glorioso, and Yury Tokpanov

Training hybrid models is hard, and papers tend to gloss over the practical engineering work that goes into building good ones. The purpose of this cookbook is to enable other technical groups to hit the ground running when building their own hybrid (SSM, Transformer, MoE) models.

For context, we at Zyphra have built the following hybrid models:
- [BlackMamba](https://arxiv.org/abs/2402.01771)
- [Zamba-7B](https://www.zyphra.com/post/zamba)
- [Zamba2-2.7B](https://www.zyphra.com/post/zamba2-small)
- [Zamba2-1.2B](TODO)

The following datasets:
- [Zyda](https://www.zyphra.com/post/zyda)

And the following engineering optimizations
- [Tree Attention](https://www.zyphra.com/post/tree-attention-topology-aware-decoding-for-long-context-attention-on-gpu-clusters)


# Introduction: How Zyphra thinks about Hybrid Models

(TODO: Someone can reign me in here if they disagree)

Dense transformer models (i.e. alternating multi-head attention (MHA) and multilayer perceptron (MLP) blocks) have dominated the DL model space for a long time. The reason for this is simple: 
1. MHA computes exact cross-sequence dependencies, and consists of GEMMs, which are easy to parallelize across many GPU SMs
2. MLPs mix the heads of MHA, and trivially boil down to GEMMs

Lots of LLM blocks (e.g. MHA, MLPs, RWKV, Mamba, KANs, xLSTM, etc) boil down to perform very similar modeling tasks. We at Zyphra intuit that the ingredients for a good LLM architecture are:
- Modeling cross-sequence dependencies (MHA, TODO BEREN/PAOLO ADD IN COMPONENTS OF OTHER BLOCKS THAT DO THIS)
- Mixing across heads (MLPs, KANs, TODO BEREN/PAOLO ADD IN COMPONENTS OF OTHER BLOCKS THAT DO THIS)

Therefore, potential LLM architectures should be evaluated on whether they:
1. Have lower FLOP and memory requirements. We believe this is most important at [inference-time](TODO: mosaicml paper), but  
2. Maintain the benefits of exact cross-sequence modeling from MHA (can be measured by proxy via [long-context reasoning](TODO) and [in-context learning](TODO)), since 

The deployment context determines which of these properties is most important, for example:
1. Massive (100B-1T+) capabilities-focused models like Grok, Claude, and ChatGPT. These models have high parameter-counts (and therefore require more training tokens to saturate) and are deployed on cloud systems with high-VRAM GPUs. This is why the low-FLOP and high-VRAM tradeoff of MoE is attractive.
2. Smaller (1B-15B) on-device special-purpose models like Zamba and Phi. These models require the lowest memory and latency at inference-time possible, and are deployed on embedded devices with strict power and memory constraints. Therefore they benefit more from SSM and hybrid architectures.

Since Zyphra seeks to build personalized on-device models, this cookbook will be focused on the practical implications of architectures falling into the smaller-model regime #2.

(TODO: Dropdown on cross-sequence dependencies, and what I mean by "exact")

# Model Architectures

## Dense Transformers

(TODO: Dense transformer pic)

## MoE Architectures

(TODO: transformer MoE and BlackMamba pics)

## SSM Architectures

(TODO: Mamba and RWKV pics)

## Hybrid Architectures

(TODO: Zamba and Jamba pics)




# Data
(TODO: Yury)

For a script on calculating the number of tokens in a TODO-formatted dataset based on a given tokenizer, see [Token Calculation](#token-calculation)

# Calculations

During the model planning phase, it's common to calculate what models will fit into a given budget of parameters, FLOPs, and inference/training memory. 

## SSM Calculations
(TODO: Quentin/Beren/Paolo)


## Transformer Calculations

For dense and MoE transformers, we recommend using the [EleutherAI cookbook](TODO) by Quentin Anthony, Hailey Schoelkopf, and Stella Biderman.

## Hybrid Calculations
(TODO: Quentin/Beren/Paolo)


## Token Calculation
(TODO: Yury/Quentin)


# Benchmarks


## Block Benchmarks and Sizing
(TODO: Quentin)


## Communication
(TODO: Quentin/Vasu)

For communication benchmarks, there are two levels of tests: 
1. Microbrenchmark-level benchmarks in C/CUDA/C++ such as [OSU-Microbenchmarks](TODO) and [NCCL-tests](TODO). These are best for checking hardware, low-level communication software and drivers, and low-level communication optimizations (e.g. [SHARP](), communication algorithm tuning, etc).
    - 
    - 
2. Framework-level benchmarks in PyTorch/Jax such as those in the [EleutherAI cookbook](TODO). These are best to ensure that framework properties (e.g. synchronization, tensor dtype handling, etc) preserve the performance of microbenchmarks, and measure performance effects of framework-level optimizations (e.g. [tensor fusion](TODO), [CUDA graphs](TODO), etc) and communication in the context of applications (e.g. communication/computation overlap)
    - 
    - 


In this cookbook, we provide framework-level benchmarks in Jax at TODO

# Training



## Annealing
(TODO: Beren/Paolo)


## Bonus: Efficient Decoding
(TODO: Vasu/Jon)