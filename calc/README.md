# Parameter and FLOP calculation walkthroughs

Here are some simple calculations as to the number of parameters and flops needed to train a given MoE model. The key difference between MoE and dense models is that each MLP layer becomes instead E parallel experts and each token is routed to a single (or many) experts. This means that the total parameter count of the model expands linearly in E but the FLOP cost to train and infer the model stays fixed. MoEs are thus presented as a possible way to infer much more efficiently than with dense models. The downsides of MoEs is that although they don’t use many FLOPs to infer, they nevertheless have a much larger amount of parameters that need to be kept in memory, and thus suffer from additional communication and storage overhead.

Let us first figure out how many parameters a given transformer has with a certain embedding dimension h, depth, l, vocab size V, and sequence length s. Let’s think step by step.

1.) First we have the embedding layer which has  parameters. If the embedding and unembedding weights are untied there are two of these.

2.) We now have the QKVO attention matrices. Each one is of dimension  for  parameters in total.

3.) Next we have the MLP layers of which we have two: mlp_in with  parameters and mlp_out with  parameters (assuming the usual expansion factor of 4. In total this gives us  parameters

4.) We have layernorm parameters (gains and biases) on each of the Q,K,V and mlp_in matrices which gives us parameters 

Now that we have the parameters per block we simply have to add them up and multiply by the number of blocks l to obtain the total number of parameters.

Then finally we add in the final layernorm with parameters and the position embedding with  parameters and we are done. This gives a final equation of:



MoE parameters

For the MoE all we are doing is adding E experts in parallel for each of the MLP blocks. This means that we simply multiply the MLP component by the number of experts to get . There is also a gating layer of size  at each block to decide which expert to route the token to. This gives a total of  parameters due to the gating. Thus the updated equation for the MoE transformer reads:



Transformer FLOPs

At the most coarse level the approximate flops for training is  where N is the amount of parameters and D is the amount of data. The reasoning for this is that each pass through the network requires approximately 1 multiply and 1 add per parameter per datapoint, and that there are effectively three passes on every step: a forward pass, a backward pass, and a weight update.

The promise of MoE is that we can only conditionally use some of the parameters per datapoint resulting in an approximate FLOPs of  if we assume that half of the parameters are from the MLP layers with MoEs (in practice for larger models this can be much more.



We can, of course be much more specific so let’s break things down further. Here we make more use of the batch size b and the sequence length s. For simplicity let’s only consider the FLOPs from the MLP layers and the attention layers since the additional parts of the transformer (the positional encoding, the embedding and unembedding layers, and the layernorms) become increasingly irrelevant for large models.

MLP layers:

For each MLP layer per token we perform  operations (2 from the multiply + accumulate). There are two MLP layers per MLP block (mlp_in and mlp_out). This gives us flops for the forward pass and thus for the backwards pass and weight update (approximately 2x as much) for a total of FLOPs for step of the MLP layers.

QKVO attention FLOPS

The dense matrix operations in attention have a very similar structure to the MLPs. We have four matrices (Q,K,V,O) which are applied per token and each matrix is  in size. This gives us FLOPs per layer per token and thus FLOPS in total for a forward pass and FLOPS for a step

Attention scores and output

To compute the attention matrix, we are multiplying together two  matrices to compute an  output. The FLOP cost of this is approximately . The output of the attention (multiplication of V by the attention scores) has an equal cost. We ignore the cost of performing the softmax although this may be nontrivial. This results in a total attention cost of  for a total step cost of  FLOPS.

Putting this together, we see that the total FLOP cost of a transformer model can be estimated as:



Naively, we thus see that the MLP cost dominates at least as long as the embedding dimension h is larger than the sequence dimension s. For large enough s however, the cost of the attention begins to dominate due to the quadratic dependence of attention on the sequence length.

MoE Flops

The MoE typically splits the MLP parameters into E parallel copies. This expands the number of parameters but does not appreciably change the FLOP cost of a step of the model, so long as only a single expert is used per token per MLP block.

FLOP budgets

The way to think about the FLOP budget is to figure out how many TFLOPS you can get per GPU running the model and then how many days you can afford to train the model for. That is, we get



Where we convert days into seconds since TFLOPs are in seconds. As an example, let’s suppose (optimistically) that we get 400 TFLOPs/H100, we have 64 H100s, and we train for 60 days, we get a budget of 132M TFLOPs.

With this number and our estimates of FLOP count for a given model and dataset size we can evaluate what model and dataset sizes are feasible to train given our FLOP budget. This gives us a space of possible models we can train. Within this space, we can use the scaling laws (if they exist) to tell us what the optimal model and dataset size would be to achieve the lowest loss.

