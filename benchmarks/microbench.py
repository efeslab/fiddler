"""Microbenchmarking for CPU offloading"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.append("../src")
from fiddler import FiddlerMixtral

def weight_copy(model, from_cpu=True):
    """Time to copy weights of an expert"""
    ret_time = []

    if from_cpu:
        expert_placeholder = copy.deepcopy(
            model.model.layers[0].block_sparse_moe.experts[0]
        ).to(model.dev)
        for i in range(32):
            model.model.layers[i].block_sparse_moe.experts[0].to("cpu")
            torch.cuda.synchronize()
            tick = time.time()
            expert_placeholder.load_state_dict(
                model.model.layers[i].block_sparse_moe.experts[0].state_dict()
            )
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            model.model.layers[i].block_sparse_moe.experts[0].to("cpu")
    else:
        expert_placeholder = copy.deepcopy(
            model.model.layers[0].block_sparse_moe.experts[0]
        ).to("cpu")
        for i in range(32):
            model.model.layers[i].block_sparse_moe.experts[0].to(model.dev)
            torch.cuda.synchronize()
            tick = time.time()
            expert_placeholder.load_state_dict(
                model.model.layers[i].block_sparse_moe.experts[0].state_dict()
            )
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
    return np.array(ret_time)


def copy_activation(model, from_cpu=True):
    """Time to copy activations"""
    ret_time = []
    if from_cpu:
        for i in range(32):
            inps = torch.randn((1, 4096), dtype=model.dtype, device="cpu")
            torch.cuda.synchronize()
            tick = time.time()
            inps = inps.to(model.dev)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps
    else:
        for i in range(32):
            inps = torch.randn((1, 4096), dtype=model.dtype, device=model.dev)
            torch.cuda.synchronize()
            tick = time.time()
            inps = inps.to("cpu")
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps
    return np.array(ret_time)


def expert_gpu(model, n_expert=1, batch_size=1):
    """Time to execute an expert at GPU"""
    ret_time = []

    # warm up
    model.model.layers[0].block_sparse_moe.experts[7].to(model.dev)
    inps = torch.randn((batch_size, 4096), dtype=model.dtype, device=model.dev)
    weights = torch.ones((batch_size, 1), dtype=model.dtype, device=model.dev)
    inps = model.model.layers[0].block_sparse_moe.experts[7](inps, weights)
    model.model.layers[0].block_sparse_moe.experts[7].to("cpu")
    del inps, weights
    torch.cuda.synchronize()

    for i in range(32):
        for j in range(n_expert):
            model.model.layers[i].block_sparse_moe.experts[j].to(model.dev)
            inps = torch.randn((batch_size, 4096), dtype=model.dtype, device=model.dev)
            weights = torch.randn((batch_size, 1), dtype=model.dtype, device=model.dev)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            tick = time.time()
            inps = model.model.layers[i].block_sparse_moe.experts[j](inps, weights)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            model.model.layers[i].block_sparse_moe.experts[j].to("cpu")
            del inps, weights
    return np.array(ret_time)


def expert_cpu(model, n_expert=1, batch_size=1, multithreading=False):
    """Time to execute an expert at CPU"""
    ret_time = []
    # warm up
    model.model.layers[0].block_sparse_moe.experts[7].to("cpu")
    inps = torch.randn((batch_size, 4096), dtype=model.dtype, device="cpu")
    weights = torch.ones((batch_size, 1), dtype=model.dtype, device="cpu")
    torch.cuda.synchronize()
    tick = time.time()
    inps = model.run_expert_at_cpu(0, 7, inps, weights)
    del inps, weights
    torch.cuda.synchronize()

    for i in range(32):
        for j in range(n_expert):
            model.model.layers[i].block_sparse_moe.experts[j].to("cpu")
            inps = torch.randn((batch_size, 4096), dtype=model.dtype, device="cpu")
            weights = torch.randn((batch_size, 1), dtype=model.dtype, device="cpu")
            torch.cuda.synchronize()
            tick = time.time()
            inps = model.run_expert_at_cpu(i, j, inps, weights)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps, weights
    return np.array(ret_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: execute at GPU (baseline), 1: offload to CPU.",
    )

    args = parser.parse_args()

    model = FiddlerMixtral(args)

    def format_output(array):
        return (
            f"mean: {np.mean(array) * 1000:.2f} ms, std: {np.std(array) * 1000:.2f} ms"
        )

    print(
        f"\n1) Weight copy, CPU -> GPU\n{format_output(weight_copy(model, from_cpu=True))}"
    )
    print(
        f"\n2) Weight copy, GPU -> CPU\n{format_output(weight_copy(model, from_cpu=False))}"
    )
    print(
        f"\n3) Activation copy, CPU -> GPU\n{format_output(copy_activation(model, from_cpu=True))}"
    )
    print(
        f"\n4) Activation copy, GPU -> CPU\n{format_output(copy_activation(model, from_cpu=False))}"
    )
    for i in [1, 2, 4, 8, 16, 32]:
        print(
            f"\n5) Execution, GPU batch={i}\n{format_output(expert_gpu(model, batch_size=i))}"
        )
    for i in [1, 2, 4, 8, 16, 32]:
        print(
            f"\n6) Execution, CPU batch={i}\n{format_output(expert_cpu(model, batch_size=i))}"
        )
