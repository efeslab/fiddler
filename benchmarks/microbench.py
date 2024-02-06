"""Microbenchmarking for CPU offloading"""
import argparse
import copy
import os
import sys
import time 
import torch 

sys.path.append("../src")

from fiddler import FiddlerMixtral

def weight_copy(model, from_cpu=True):
    """Time to copy weights of an expert"""
    sum_time = 0

    if from_cpu:
        expert_placeholder = copy.deepcopy(
            model.model.layers[0].block_sparse_moe.experts[0]
        ).to(model.dev)
        for i in range(32):
            model.model.layers[i].block_sparse_moe.experts[0].to('cpu')
            torch.cuda.synchronize()
            tick = time.time()
            expert_placeholder.load_state_dict(
                model.model.layers[i].block_sparse_moe.experts[0].state_dict()
            )
            torch.cuda.synchronize()
            sum_time += time.time() - tick
            model.model.layers[i].block_sparse_moe.experts[0].to('cpu')
    else:
        expert_placeholder = copy.deepcopy(
            model.model.layers[0].block_sparse_moe.experts[0]
        ).to('cpu')
        for i in range(32):
            model.model.layers[i].block_sparse_moe.experts[0].to(model.dev)
            torch.cuda.synchronize()
            tick = time.time()
            expert_placeholder.load_state_dict(
                model.model.layers[i].block_sparse_moe.experts[0].state_dict()
            )
            torch.cuda.synchronize()
            sum_time += time.time() - tick
    return sum_time / 32

def copy_activation(model, from_cpu=True):
    """Time to copy activations"""
    sum_time = 0
    if from_cpu:
        for i in range(32):
            inps = torch.randn((1, 4096), dtype=model.dtype, device='cpu')
            torch.cuda.synchronize()
            tick = time.time()
            inps = inps.to(model.dev)
            torch.cuda.synchronize()
            sum_time += time.time() - tick
            del inps 
    else:
        for i in range(32):
            inps = torch.randn((1, 4096), dtype=model.dtype, device=model.dev)
            torch.cuda.synchronize()
            tick = time.time()
            inps = inps.to('cpu')
            torch.cuda.synchronize()
            sum_time += time.time() - tick
            del inps 
    return sum_time / 32

def expert_gpu(model, n_expert=1, batch_size=1):
    """Time to execute an expert at GPU"""
    sum_time = 0
    for i in range(32):
        for j in range(n_expert):
            model.model.layers[i].block_sparse_moe.experts[j].to(model.dev)
            inps = torch.randn((batch_size, 4096), dtype=model.dtype, device=model.dev)
            weights = torch.ones((batch_size, 1), dtype=model.dtype, device=model.dev)
            torch.cuda.synchronize()
            tick = time.time()
            inps = model.model.layers[i].block_sparse_moe.experts[j](inps, weights)
            torch.cuda.synchronize()
            sum_time += time.time() - tick
            model.model.layers[i].block_sparse_moe.experts[j].to('cpu')
            del inps, weights
    return sum_time / 32

def expert_cpu(model, n_expert=1, batch_size=1, multithreading=False):
    """Time to execute an expert at CPU"""
    sum_time = 0
    for i in range(32):
        for j in range(n_expert):
            model.model.layers[i].block_sparse_moe.experts[j].to('cpu')
            inps = torch.randn((batch_size, 4096), dtype=model.dtype, device='cpu')
            weights = torch.ones((batch_size, 1), dtype=model.dtype, device='cpu')
            torch.cuda.synchronize()
            tick = time.time()
            inps = model.run_expert_at_cpu(i, j, inps, weights)
            torch.cuda.synchronize()
            sum_time += time.time() - tick
            del inps, weights
    return sum_time / 32

    # # 7: execute 2 experts at CPU sequentially
    # sum_time = 0
    # for i in range(32):
    #     model.model.layers[i].block_sparse_moe.experts[0].to('cpu')
    #     model.model.layers[i].block_sparse_moe.experts[1].to('cpu')
    #     inps0 = torch.randn((1, 4096), dtype=model.dtype, device='cpu')
    #     inps1 = torch.randn((1, 4096), dtype=model.dtype, device='cpu')
    #     weights0 = torch.ones((1, 1), dtype=model.dtype, device='cpu')
    #     weights1 = torch.ones((1, 1), dtype=model.dtype, device='cpu')
    #     torch.cuda.synchronize()
    #     tick = time.time()
    #     inps0 = model.run_expert_at_cpu(i, 0, inps0, weights0)
    #     inps1 = model.run_expert_at_cpu(i, 1, inps1, weights1)
    #     torch.cuda.synchronize()
    #     sum_time += time.time() - tick
    #     del inps0, inps1, weights0, weights1
    # print(f'7) Execution, CPU (2 experts, sequential): {sum_time / 32 * 1000} ms')

    # # 8: execute 2 experts at CPU with multithreading
    # sum_time = 0
    # for i in range(32):
    #     model.model.layers[i].block_sparse_moe.experts[0].to('cpu')
    #     model.model.layers[i].block_sparse_moe.experts[1].to('cpu')
    #     inps0 = torch.randn((1, 4096), dtype=model.dtype, device='cpu')
    #     inps1 = torch.randn((1, 4096), dtype=model.dtype, device='cpu')
    #     weights0 = torch.ones((1, 1), dtype=model.dtype, device='cpu')
    #     weights1 = torch.ones((1, 1), dtype=model.dtype, device='cpu')
    #     torch.cuda.synchronize()
    #     tick = time.time()
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future_0 = executor.submit(
    #             model.run_expert_at_cpu,
    #             i, # layer id
    #             0, # expert id
    #             inps0, 
    #             weights0,
    #         )
    #         future_1 = executor.submit(
    #             model.run_expert_at_cpu,
    #             i, # layer id
    #             1, # expert id
    #             inps1, 
    #             weights1,
    #         )
    #     inps0 = future_0.result()
    #     inps1 = future_1.result()
    #     torch.cuda.synchronize()
    #     sum_time += time.time() - tick
    #     del inps0, inps1, weights0, weights1
    # print(f'8) Execution, CPU (2 experts, parallel): {sum_time / 32 * 1000} ms')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    parser.add_argument(
        '--model', type=str, default='mistralai/Mixtral-8x7B-v0.1',
        help='Switch model to load; pass `mistralai/Mixtral-8x7B-v0.1`.'
    )
    parser.add_argument(
        '--cpu-offload', type=int, default=1, choices=[0, 1],
        help='0: exeute at GPU (baseline), 1: offload to CPU.'
    )
    parser.add_argument(
        '--input', type=str, default='Mistral AI is a',
        help='Input text to generate.'
    )
    parser.add_argument(
        '--n-token', type=int, default=20,
        help='Number of tokens to generate.',
    )

    args = parser.parse_args() 

    model = FiddlerMixtral(args)
    
    print(f'1) Weight copy, CPU -> GPU: {weight_copy(model, from_cpu=True) * 1000} ms')
    print(f'2) Weight copy, GPU -> CPU: {weight_copy(model, from_cpu=False) * 1000} ms')
    print(f'3) Copy activation, CPU -> GPU: {copy_activation(model, from_cpu=True) * 1000} ms')
    print(f'4) Copy activation, GPU -> CPU: {copy_activation(model, from_cpu=False) * 1000} ms')
    for i in range(1, 10 + 1):
        print(f'5) Execution, GPU batch={i}: {expert_gpu(model, batch_size=i) * 1000} ms')
    for i in range(1, 10 + 1):
        print(f'6) Execution, CPU batch={i}: {expert_cpu(model, batch_size=i) * 1000} ms')
