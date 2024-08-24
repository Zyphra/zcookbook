# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from mamba_ssm import Mamba2, Mamba

from benchmarks.computation.utils import time_fwd_bwd, ns_per_param

# # To Print out location of library pointers
# import inspect
# print( inspect.getmodule(Mamba) )
# print( inspect.getmodule(Mamba2) )

# def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
#     assert mode in ["fwd", "bwd", "fwd_bwd"]
#     f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
#     return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

# def flops_mamba(batch, seqlen, d_model, d_state, expand, mode="fwd"):
#     """
#     Not certain this calculation is entirely correct for Mamba.
#     Ref: albertfgu comment in https://github.com/state-spaces/mamba/issues/110
    
#     Probably it's different for Mamba2 too.
#     """
#     assert mode in ["fwd", "bwd", "fwd_bwd"]
#     f = 9 * d_state * d_model * seqlen * batch
#     return f if mode == "fwd" else (2 * f if mode == "bwd" else 3 * f)


# def efficiency(flop, time):
#     return (flop / time / 10**12) if not math.isnan(time) else 0.0

torch_profiler_flag = False

repeats = 30
device = 'cuda'
dtype = torch.float16 # torch.float16 or torch.bfloat16
#print(f"dtype = {dtype}")

bs_seqlen_vals = [(2, 64)]
d_model_vals = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
d_state_vals = [64]
expand_vals = [2]


methods = (["Flash2-Mamba"])
#methods = (["Flash2-Mamba", "Flash2-Mamba2"])

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

if torch_profiler_flag:
    schedule = torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2
    )
    prof = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            './'
        ),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
        with_stack=True,
    )
    prof.start()


for batch, seqlen in bs_seqlen_vals:
    for d_model in d_model_vals:
        for d_state in d_state_vals:
            for expand in expand_vals:

                config = (batch, d_model, d_state, expand)

                # Run Mamba through Flash 2
                input = torch.randn(batch, seqlen, d_model, device=device, dtype=dtype, requires_grad=True)
                model = Mamba(d_model=d_model, d_state=d_state, expand=expand, device=device, dtype=dtype)

                num_params = int(0)
                for parameter in model.parameters():
                    num_params += np.prod(parameter.shape)

                f, b = time_fwd_bwd(model, input, repeats=repeats, verbose=False)
                time_f[config, "Flash2-Mamba"] = f
                time_b[config, "Flash2-Mamba"] = b

                if torch_profiler_flag:
                    prof.step()


                print(f"\n### dtype = {dtype}, d_model={d_model}, d_state={d_state}, expand={expand}, batch_size={batch}, seqlen={seqlen}###")
                for method in methods:
                    time_f_b[config, method] = time_f[config, method] + time_b[config, method]

                    speed_f[config, method] = ns_per_param(time_f[config, method], num_params)
                    speed_b[config, method] = ns_per_param(time_b[config, method], num_params)
                    speed_f_b[config, method] = ns_per_param(time_f_b[config, method], num_params)
                    
                    print(
                        f"{method} fwd: {speed_f[config, method]:.5f} ns/param, "
                        f"bwd: {speed_b[config, method]:.5f} ns/param, "
                        f"fwd + bwd: {speed_f_b[config, method]:.5f} ns/param"
                    )


if torch_profiler_flag:
    prof.stop()


#import IPython; IPython.embed()


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
