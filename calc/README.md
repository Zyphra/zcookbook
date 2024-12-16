# Parameter and FLOP calculation walkthroughs

Here we present a walk-through of how to compute the parameters and FLOPs for a variety of architectures such as standard transformers, MoEs, Mamba, and Mamba2.

## Dense Transformer Parameters

Let us first figure out how many parameters a given transformer has with a certain embedding dimension h, depth, l, vocab size V, and sequence length s. To do so, we walk systematically through every parameter.

1.) First we have the embedding which has $$V \times h$$  parameters. If the embedding and unembedding weights are untied there are two of these, but let's assume they're tied for our analysis.

2.) We now have the QKVO attention matrices. Each one is of dimension $$h \times h$$ for $$4h^2$$ parameters in total.

3.) Next we have the MLP layers of which we have two: mlp_in with $$h \times 4h$$ parameters and mlp_out with $$4h \times h$$ parameters (assuming the usual expansion factor of 4). In total, this gives us $$8h^2$$ parameters for th

4.) We have layernorm parameters (gains and biases) on each of the Q,K,V and mlp_in matrices which gives us $$8h$$ parameters 

Now that we have the parameters per layer we simply have to multiply by the number of layers $$L$$ to obtain the total number of parameters contributed from attention, MLPs, and per-layer norms.

Then finally we add in the final layernorm with $$2h$$ parameters and the position embedding with $$sh$$ parameters and we are done. This gives a final equation of:

$$\begin{align}
\text{Total Dense Transformer Parameters} = Vh + sh + L(12h^2 + 8h) + 2h
\end{align}$$

## MoE Transformer Parameters

The key difference between MoE and dense models is that each MLP block instead becomes $$E$$ parallel experts and each token is routed to a single (or many) experts (See [Model Architectures](https://github.com/Zyphra/zcookbook/blob/main/README.md#model-architectures)). This means that the total parameter count of the model expands linearly in $$E$$ but the FLOP cost to train and infer the model stays fixed at that of the forward pass of the original dense. MoEs are thus a possible way to infer much more efficiently than with dense models. The downsides of MoEs is that although they don’t use many FLOPs to infer, they nevertheless have a much larger amount of parameters that need to be kept in memory, and thus suffer from additional communication and storage overhead.

Since a MoE is otherwise identical to a transformer except each MLP is copied $$E$$ times, this means that we simply multiply the MLP component $$8Lh^2$$ by the number of experts to get $$8Elh^2$$. There is also a routing layer of size $$E \times h$$ at each block to decide which expert to route the token to. This gives a total of $$EhL$$ parameters due to the router. Thus the updated equation for the MoE transformer reads:

$$\begin{align}
\text{Total Transformer MoE Parameters} = Vh + sh + L(4h^2 + 8Eh^2 + Eh + 8h) + 2h
\end{align}$$

## Transformer FLOPs

At the most coarse level the approximate flops for training is $$6 \times N \times D$$ where N is the amount of parameters and D is the amount of data. The reasoning for this is that each pass through the network requires approximately 1 multiply and 1 add per parameter per datapoint, and that there are effectively three passes on every step: a forward pass, a backward pass, and a weight update.

The promise of MoE is that we can only conditionally use some of the parameters per datapoint resulting in an approximate FLOPs of $$6 \times \frac{1}{2}(N + \frac{N}{E}) \times D$$  if we assume that half of the parameters are from the MLP layers with MoEs (in practice for larger models this can be much more).

We can, of course be much more specific so let’s break things down further. Here we make more use of the batch size b and the sequence length s. For simplicity let’s only consider the FLOPs from the MLP layers and the attention layers since the additional parts of the transformer (the positional encoding, the embedding and unembedding layers, and the layernorms) become increasingly irrelevant for large models.

### MLP Block FLOPs

For each MLP layer per token we perform $$2 \times 4h \times h$$ operations (2 from the multiply + accumulate). There are two MLP layers per MLP block (mlp_in and mlp_out). This gives us $$16bsLh^2$$ flops for the forward pass and thus $$32bLsh^2$$ for the backwards pass and weight update (approximately 2x as much) for a total of $$48bLsh^2$$ FLOPs for step of the MLP layers.

### QKVO Attention FLOPs

The dense matrix operations in attention have a very similar structure to the MLPs. We have four matrices (Q,K,V,O) which are applied per token and each matrix is $$h \times h$$  in size. This gives us $$4 \times 2 \times h^2$$ FLOPs per layer per token and thus $$8bsLh^2$$ FLOPs in total for a forward pass and $$24bsLh^2$$ FLOPs for a step.

### Attention Scores and Output FLOPs

To compute the attention matrix, we are multiplying together two $b \times s \times h$ matrices to compute an $$b \times s \times s$$ output. The FLOP cost of this is approximately $$2 \times b \times h \times s \times s$$ . The output of the attention (multiplication of V by the attention scores) has an equal cost. We ignore the cost of performing the softmax although this may be nontrivial. This results in a total attention cost of $$4bhs^2$$ for a total step cost of $$12bhs^2$$ FLOPs.

Putting this together, we see that the total FLOP cost of a transformer model can be estimated (by only accounting for the FLOPs of attention and MLP blocks) as:

$$\begin{align}
\text{Transformer FLOPs per Step} = 12bLhs^2 + 72bLsh^2
\end{align}$$

Naively, we thus see that the MLP cost dominates at least as long as the embedding dimension h is larger than the sequence dimension s. For large enough s however, the cost of the attention begins to dominate due to the quadratic dependence of attention on the sequence length.

## MoE Flops

The MoE typically splits the MLP parameters into E parallel copies. This expands the number of parameters but does not appreciably change the FLOP cost of a step of the model, so long as only a single expert is used per token per MLP block.

## Mamba Parameters

We now consider computing the parameter and flops of a single Mamba1 and Mamba2 block. First we handle parameters. For Mamba1 these calculations can also be found in the Appendix of our [BlackMamba paper](https://arxiv.org/abs/2402.01771).

### Mamba1 Parameters

This refers to the original Mamba block as introduced [here](https://arxiv.org/pdf/2312.00752). A Mamba block takes input from the residual stream of size $D \times S$ where D is the embedding dimension and L is the sequence length. The mamba block also has an internal expansion factor (typically of 2) where it operates in a larger internal embedding dimension we denote I. The Mamba block also contains an internal causal-convolutional layer with kernel width C and an internal SSM state of size S. Finally, the Mamba layer has a dt projection which controls the size of the small MLP which is used to set the token-wise time-constant for the SSM.

The Mamba layer begins with two input projections of size $$D \times I$$ which map the residual stream input into the inner dimension. There are two projections one for the SSM input itself and secondly for the gate input. This gives a total of $$2ID$$ in-projector parameters. After the in-projector there is a convolutional layer prior to the SSM. This requires $$C \times I$$ parameters plus an additional $$I$$ parameters for the convolutional bias. 

This is then followed by the matrices producing the A,B, and C matrices of the SSM (similar to the QKV matrices of attention). Each of these matrices is of size $$I \times S$$ resulting in $$3IS$$ parameters. Additionally, there is the dt projector which consists of $$2 \times dt \times I$$ as well as a dt bias and the D bias vector both of length $$I$$. Finally, there is the SSM outprojector of size $$I \times D$$ which maps back to the embedding dimension of the residual stream, as well as the input layernorm which contains $2 \times D$ parameters since its gain and bias parameters are both of shape $$D$$. 

Putting this all together, we obtain the following count of total parameters:

$$\begin{align}
\text{Total parameters} = 3ID + 2I(S + dt + \frac{C}{2}) + I + 2D
\end{align}$$

## Mamba2 Parameters

The Mamba2 block (introduced [here](https://arxiv.org/abs/2405.21060)) introduces a few modifications to the Mamba1 block to improve flop efficiently. These primarily consist in making the A matrix scalar instead of diagonal and making the B,C,dt matrices depend directly on the input from the residual stream instead of first passing through the convolution. It also introduces the notion of heads, similar to attention heads, and groups which are similar to the repeated heads in GQA. We denote the number of groups as G and the number of heads as H.

We begin by computing the in-projector as before. This consists of the input and gate projections of shape $$D \times I$$ each as well as the B and C matrices of shape $$S \times G$$ each and the dt projection of shape $$H$$ (dt is now a scalar per Mamba head). There are also the A and D matrices of shape $$H$$.

The in-projector is followed by the convolution which is applied to the x, B, and C matrices. The total parameters for the convolution are thus $$I + 2GS \times C$$.  and the convolutional bias of shape $$I + 2GS$$. Following the convolution, unlike Mamba1 there is also an additional SSM internal layernorm which utilizes $$2I$$ parameters. Following the SSM, there is then the out-projector matrix which is of size $$I \times D$$. Putting this all together, we obtain an expression for the parameters in a Mamba2 layer as:

$$\begin{align}
\text{Total parameters} = 3ID + 2DGS + DH + 2H + (I + 2GS)(1 + C) + I
\end{align}$$

## Mamba FLOPs

In general, given two matrices $$A^{K \times M}$$ and $$B^{M \times J}$$ the total flops of computing their matrix product is $$2KMJ$$ where the 2 comes from the fact that there is both a multiple and an addition operation. 

Let us consider the in and out projectors of Mamba. These are matrices of shape $I \times D$ being multipled with input of shape $B \times L \times D$ and there are three such matrix multiplications $$W_x, W_z, W_y$$ resulting in $$6BLID$$ FLOPs. Next is the convolution which can be treated as a single $$I \times C$$ matrix multiply requiring $$2BLIC$$ FLOPs. 

Now, we turn to the SSM block itself. We first compute the input-dependent B and C matrices requiring a matrix multiply of shape $$I \times H$$ each thus resulting in $$4BLIS$$ FLOPs. The A matrix is not multiplied by the input but goes through an elementwise transform costing $$IS$$ FLOPs. The dt projection first goes through an elementwise operation of order $$BLIdt$$ FLOPs.
Next, the discretization. The A matrix is multiplied by the dt vector resulting, costing $$BLIS$$ FLOPs. The B matrix is multiplied by the dt costing $$2BLIS$$ FLOPs. The SSM linear state space step itself is just a matrix multiply and add so costs $$2BLIS$$ FLOPs, and then the output projection using the C matrix also costs $$2BLIS$$ FLOPs. Finally there is the out-projector which costs $$2BLEI$$ FLOPs Putting this all together, we obtain the following expression:

$$\begin{align}
\text{Total FLOPs} = BLI(6D + 2C + 8IS + 2E + dt) + IS
\end{align}$$

## Mamba2 FLOPs

Computing the flops of a Mamba2 block involves going through a similar exercise. First we consider the much-enlarged in-projector of Mamba2 which is of shape $$(2I + 2GS + H) \times D$$ which is multiplied by the embedding input of size $$B \times L \times D$$. This results in $$2BL(2I + 2GS + H)D$$ FLOPs. The in-projector is then split and only the xBC matrix is passed through the conv at the FLOP cost of $$2BL(I + 2GS)C$$. Following the conv there is the computation of the ssm state matrices and multiplication by dt which costs $$2BLIS$$ FLOPs and the SSM computation itself which also costs $$2BLIS$$ FLOPs. Finally, there is the multiplication by the C matrix which costs $$2BLIS$$ and the multiplication by the gate which costs $$BLI$$, and finally the multiplication by the out-projector costing $$2BLIE$$. Putting this all together we obtain:

$$\begin{align}
\text{Total FLOPs} = BL\Big(4ID + 2GSD + 2HD + 2IC + 4GSC + 6IS + 2SD + I + 2IE + D\Big)
\end{align}$$





## FLOP Budgets

The way to think about the FLOP budget is to figure out how many TFLOPs you can get per GPU running the model and then how many days you can afford to train the model for. That is, we get

$$\begin{align}
\text{FLOP budget} = \text{TFLOPs per GPU} \times \text{NUM GPUs} \times \text{Days} \times 24 \times 60 \times 60
\end{align}$$

Where we convert days into seconds since TFLOPs are in seconds. As an example, let’s suppose (optimistically) that we get 400 TFLOPs/H100, we have 64 H100s, and we train for 60 days, we get a budget of 132M TFLOPs.

With this number and our estimates of FLOP count for a given model and dataset size we can evaluate what model and dataset sizes are feasible to train given our FLOP budget. This gives us a space of possible models we can train. Within this space, we can use the scaling laws (if they exist) to tell us what the optimal model and dataset size would be to achieve the lowest loss.


## Token Calculation

We provide a script at https://github.com/Zyphra/cookbook/tree/main/calc/data/tokenize_and_count.py that tokenizes text data from a Hugging Face dataset, calculates the total number of tokens, and optionally saves the tokenized dataset.

### Requirements

- Python 3.6+
- transformers
- datasets

Install the required packages:

```
pip install transformers datasets
```

### Usage

Run the script from the command line with the following arguments:

```
python tokenize_and_count.py --hf-path <dataset_path> --hf-tokenizer <tokenizer_path> [OPTIONS]
```


#### Required Arguments:

- `--hf-path`: Path of the Hugging Face dataset
- `--hf-tokenizer`: Path of the Hugging Face tokenizer

#### Optional Arguments:

- `--hf-dir`: Directory in the Hugging Face dataset (default: None)
- `--key`: Name of the column that contains text to tokenize (default: 'text')
- `--save-path`: Folder to save processed Hugging Face dataset to (default: None)
- `--num-proc`: Number of processes for Hugging Face processing (default: 1)

### Example

```
python tokenize_and_count.py --hf-path "dataset/my_dataset" --hf-tokenizer "bert-base-uncased" --key "content" --save-path "./tokenized_dataset" --num-proc 4
```


This command will:
1. Load the dataset from "dataset/my_dataset"
2. Use the "bert-base-uncased" tokenizer
3. Tokenize the "content" column of the dataset
4. Use 4 processes for parallel processing
5. Save the tokenized dataset to "./tokenized_dataset"

The script will output the total number of tokens in the dataset and save the tokenized dataset if a save path is provided.
