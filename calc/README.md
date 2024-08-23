# Parameter and FLOP calculation walkthroughs

Here we present a walk-through of how to compute the parameters and FLOPs for a variety of architectures such as standard transformers, MoEs, Mamba and Mamba2.

## Transformer parameters

Let us first figure out how many parameters a given transformer has with a certain embedding dimension h, depth, l, vocab size V, and sequence length s. To do so, we walk systematically through every parameter.

1.) First we have the embedding layer which has $$V \times h$$  parameters. If the embedding and unembedding weights are untied there are two of these.

2.) We now have the QKVO attention matrices. Each one is of dimension $$h \times h$$ for $$4h^2$$  parameters in total.

3.) Next we have the MLP layers of which we have two: mlp_in with $$h \times 4h$$ parameters and mlp_out with $$4h \times h$$ parameters (assuming the usual expansion factor of 4. In total, this gives us $$8h^2$$ parameters.

4.) We have layernorm parameters (gains and biases) on each of the Q,K,V and mlp_in matrices which gives us $$8h$$ parameters 

Now that we have the parameters per block we simply have to add them up and multiply by the number of blocks l to obtain the total number of parameters.

Then finally we add in the final layernorm with $$2h$$ parameters and the position embedding with $$sh$$ parameters and we are done. This gives a final equation of:

$$\begin{align}
\text{total params} = Vh + sh + 12lh^2 + 8hl + 2h
\end{align}$$

## MoE parameters

The key difference between MoE and dense models is that each MLP layer instead becomes E parallel experts and each token is routed to a single (or many) experts. This means that the total parameter count of the model expands linearly in E but the FLOP cost to train and infer the model stays fixed at that of the forward pass of the original dense. MoEs are thus a possible way to infer much more efficiently than with dense models. The downsides of MoEs is that although they don’t use many FLOPs to infer, they nevertheless have a much larger amount of parameters that need to be kept in memory, and thus suffer from additional communication and storage overhead.

Since a MoE is otherwise identical to a transformer except each MLP is copied E times, this means that we simply multiply the MLP component $$8lh^2$$ by the number of experts to get $$8Elh^2$$. There is also a routing layer of size $$E \times h$$ at each block to decide which expert to route the token to. This gives a total of $$Ehl$$ parameters due to the router. Thus the updated equation for the MoE transformer reads:

$$\begin{align}
\text{total params} = Vh + sh + 4hl^2 + 8ELh^2 + Ehl + 8hl + 2h
\end{align}$$

## Transformer FLOPs

At the most coarse level the approximate flops for training is $$6 \times N \times D$$ where N is the amount of parameters and D is the amount of data. The reasoning for this is that each pass through the network requires approximately 1 multiply and 1 add per parameter per datapoint, and that there are effectively three passes on every step: a forward pass, a backward pass, and a weight update.

The promise of MoE is that we can only conditionally use some of the parameters per datapoint resulting in an approximate FLOPs of $$6 \times \frac{1}{2}(N + \frac{N}{E}) \times D$$  if we assume that half of the parameters are from the MLP layers with MoEs (in practice for larger models this can be much more).

We can, of course be much more specific so let’s break things down further. Here we make more use of the batch size b and the sequence length s. For simplicity let’s only consider the FLOPs from the MLP layers and the attention layers since the additional parts of the transformer (the positional encoding, the embedding and unembedding layers, and the layernorms) become increasingly irrelevant for large models.

MLP layers:

For each MLP layer per token we perform $$2 \times 4h \times h$$ operations (2 from the multiply + accumulate). There are two MLP layers per MLP block (mlp_in and mlp_out). This gives us $$16bslh^2$$ flops for the forward pass and thus $$32blsh^2$$ for the backwards pass and weight update (approximately 2x as much) for a total of $$48blsh^2$$ FLOPs for step of the MLP layers.

QKVO attention FLOPS

The dense matrix operations in attention have a very similar structure to the MLPs. We have four matrices (Q,K,V,O) which are applied per token and each matrix is $$h \times h$$  in size. This gives us $$4 \times 2 \times h^2$$ FLOPs per layer per token and thus $$8bslh^2$$ FLOPS in total for a forward pass and $$24bslh^2$$ FLOPS for a step.

Attention scores and output

To compute the attention matrix, we are multiplying together two $b \times s \times h$$ matrices to compute an $$b \times s \times s$$ output. The FLOP cost of this is approximately $$2 \times b \times h \times s \times s$$ . The output of the attention (multiplication of V by the attention scores) has an equal cost. We ignore the cost of performing the softmax although this may be nontrivial. This results in a total attention cost of $$4bhs^2$$ for a total step cost of $$12bhs^2$$ FLOPS.

Putting this together, we see that the total FLOP cost of a transformer model can be estimated as:

$$\begin{align}
\text{FLOPS per step} = 12bhs^2 + 72blsh^2
\end{align}$$

Naively, we thus see that the MLP cost dominates at least as long as the embedding dimension h is larger than the sequence dimension s. For large enough s however, the cost of the attention begins to dominate due to the quadratic dependence of attention on the sequence length.

## MoE Flops

The MoE typically splits the MLP parameters into E parallel copies. This expands the number of parameters but does not appreciably change the FLOP cost of a step of the model, so long as only a single expert is used per token per MLP block.

## Mamba parameters

We now consider computing the parameter and flops of a single Mamba1 and Mamba2 block. First we handle parameters. For Mamba1 these calculations can also be found in the Appendix of our [BlackMamba paper](https://arxiv.org/abs/2402.01771).

### Mamba1 parameters

This refers to the original Mamba block as introduced [here](https://arxiv.org/pdf/2312.00752). A Mamba block takes input from the residual stream of size $D \times S$ where D is the embedding dimension and L is the sequence length. The mamba block also has an internal expansion factor (typically of 2) where it operates in a larger internal embedding dimension we denote I. The Mamba block also contains an internal causal-convolutional layer with kernel width C and an internal SSM state of size S. Finally, the Mamba layer has a dt projection which controls the size of the small MLP which is used to set the token-wise time-constant for the SSM.

The Mamba layer begins with two input projections of size $$D \times I$$ which map the residual stream input into the inner dimension. There are two projections one for the SSM input itself and secondly for the gate input. This gives a total of $$2ID$$ in-projector parameters. After the in-projector there is a convolutional layer prior to the SSM. This requires $$C \times I$$ parameters plus an additional $$I$$ parameters for the convolutional bias. 

This is then followed by the matrices producing the A,B, and C matrices of the SSM (similar to the QKV matrices of attention). Each of these matrices is of size $$I \times S$$ resulting in $$3IS$$ parameters. Additionally, there is the dt projector which consists of $$2 \times dt \times I$$ as well as a dt bias and the D bias vector both of length $$I$$. Finally, there is the SSM outprojector of size $$I \times D$$ which maps back to the embedding dimension of the residual stream, as well as the input layernorm which contains $2 \times D$$ parameters since its gain and bias parameters are both of shape $$D$$. 

Putting this all together, we obtain the following count of total parameters:

$$\begin{align}
\text{Total parameters} = 3ID + 2I(S + dt + \frac{C}{2}) + I + 2D
\end{align}$$

## Mamba2 parameters

The Mamba2 block (introduced [here](https://arxiv.org/abs/2405.21060)) introduces a few modifications to the Mamba1 block to improve flop efficiently. These primarily consist in making the A matrix scalar instead of diagonal and making the B,C,dt matrices depend directly on the input from the residual stream instead of first passing through the convolution. It also introduces the notion of heads, similar to attention heads, and groups which are similar to the repeated heads in GQA. We denote the number of groups as G and the number of heads as H.

We begin by computing the in-projector as before. This consists of the input and gate projections of shape $$D \times I$$ each as well as the B and C matrices of shape $$S \times G$$ each and the dt projection of shape $$H$$ (dt is now a scalar per Mamba head). There are also the A and D matrices of shape $$H$$.

The in-projector is followed by the convolution which is applied to the x, B, and C matrices. The total parameters for the convolution are thus $$I + 2GS \times C$$.  and the convolutional bias of shape $$I + 2GS$$. Following the convolution, unlike Mamba1 there is also an additional SSM internal layernorm which utilizes $$2I$$ parameters. Following the SSM, there is then the out-projector matrix which is of size $$I \times D$$. Putting this all together, we obtain an expression for the parameters in a Mamba2 layer as:

$$\begin{align}
\text{Total parameters} = 3ID + 2DGS + DH + 2H + (I + 2GS)(1 + C) + I
\end{align}$$

## FLOP budgets

The way to think about the FLOP budget is to figure out how many TFLOPS you can get per GPU running the model and then how many days you can afford to train the model for. That is, we get

$$\begin{align}
\text{FLOP budget} = \text{TFLOPs per GPU} \times \text{NUM GPUs} \times \text{Days} \times 24 \times 60 \times 60
\end{align}$$

Where we convert days into seconds since TFLOPs are in seconds. As an example, let’s suppose (optimistically) that we get 400 TFLOPs/H100, we have 64 H100s, and we train for 60 days, we get a budget of 132M TFLOPs.

With this number and our estimates of FLOP count for a given model and dataset size we can evaluate what model and dataset sizes are feasible to train given our FLOP budget. This gives us a space of possible models we can train. Within this space, we can use the scaling laws (if they exist) to tell us what the optimal model and dataset size would be to achieve the lowest loss.

