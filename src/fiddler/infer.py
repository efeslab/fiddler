import os
import argparse
import copy
import time
import random
import concurrent.futures

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from torch.autograd import profiler

class FiddlerMixtral():
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device('cuda:0')
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=self.dtype,
            # load_in_8bit=True,
            # device_map='cpu',
            use_cache=True,
        )
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.cpu_offload = args.cpu_offload

        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(f'Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}')

        # prof_data = np.zeros((self.n_layer, self.n_expert), dtype=int)
        # with open('cnt-expert.txt', 'r') as f:
        #     for line in f:
        #         i_layer, i_expert, cnt = map(int, line.strip().split('-'))
        #         prof_data[i_layer, i_expert] = cnt 
        # popular_experts = np.argsort(prof_data.flatten())[::-1]
        # for i in range(n_expert_on_gpu):
        #     i_layer, i_expert = divmod(popular_experts[i], self.n_expert)
        #     self.expert_loc[i_layer, i_expert] = 1
        for i_layer in range(self.n_layer):
            for i_expert in range(self.n_expert):
                if i_layer + i_expert * self.n_layer < n_expert_on_gpu:
                    self.expert_loc[i_layer, i_expert] = 1
        print(self.expert_loc)
    
        self.bring_expert_to_gpu()

        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            # only model.layers[i].block_sparse_moe.experts is on CPU
    
    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)
    
    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1

    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(p.numel() for p in self.model.layers[0].block_sparse_moe.experts[0].parameters())
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev)
        return int((free_mem) // (n_param * 2))

    def generate(self, text, output_token=20, input_token=None):
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        input_ids, position_ids = self.tokenize(text)

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]
        
        tick = time.time()
        is_decode = False 
        prefill_time, decode_time = 0, 0
        for i_token in range(output_token):
            # tick = time.time()
            print(self.tokenizer.decode(input_ids[0, :]))
            logits = self.mixtral_forward(
                input_ids, 
                position_ids,
                is_decode,
            )
            # print('Time:', time.time() - tick)

            logits = logits.to('cpu')

            output = torch.argmax(logits, dim=-1)
            self.past_key_values_length += output.shape[-1]
            input_ids = output[:, -1].unsqueeze(0).to(self.dev)
            position_ids = torch.arange(self.past_key_values_length, self.past_key_values_length + 1, dtype=torch.long, device=self.dev)
            position_ids = position_ids.unsqueeze(0).view(-1, 1)
            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
        decode_time = time.time() - tick
        return prefill_time, decode_time, self.cnt_expert_hit / self.cnt_expert_all
    
    def tokenize(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.dev)
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=self.dev)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
        return input_ids, position_ids
    
    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps)

        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape
            
            inps_residual = inps
            inps = layer.input_layernorm(inps)
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)

            inps = inps.view(-1, hidden_dim)
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1) 
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            if self.cpu_offload == 0:
                # baseline: do everything at GPU
                expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=8).permute(2, 1, 0)
                
                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        # print(f"Expert {i_expert}: has no tokens")
                        continue 
                    
                    # torch.cuda.synchronize()
                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if not is_cuda:
                        self.expert_placeholder.load_state_dict(experts[i_expert].state_dict())
                        current_state = self.expert_placeholder(current_state, routing_weights[top_2_list, idx_list, None])
                    else:
                        current_state = experts[i_expert](current_state, routing_weights[top_2_list, idx_list, None])
                    inps_after_experts.index_add_(0, top_2, current_state.to(inps.dtype))

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to('cpu', non_blocking=True)
                    
                    # end of one expert
            
            elif not is_decode:
                # prefill stage with offloading
                expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=8).permute(2, 1, 0)

                # first, calculate the number of tokens for each expert
                idxs, top_2s = [], []
                cost_per_expert = np.zeros((len(experts), 2), dtype=float) # 0: CPU, 1: GPU

                # TODO: find this value based on device config 
                cost_at_cpu = 7
                cost_at_gpu = 70
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])
                    idxs.append(idx)
                    top_2s.append(top_2)
                    # expected latency at CPU: number of token * cost_at_cpu
                    # expected latency at GPU: cost_at_gpu (constant)
                    cost_per_expert[i_expert, 0] = top_2.shape[0] * cost_at_cpu
                    cost_per_expert[i_expert, 1] = cost_at_gpu
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # if the expert is in GPU, the latency at GPU is approximately 0
                        cost_per_expert[i_expert, 1] = 0
                        self.cnt_expert_hit += top_2.shape[0]
                    self.cnt_expert_all += top_2.shape[0]
                
                # second, partition experts processing between CPU and GPU so that we can minimize:
                # max(sum of cost at CPU, sum of cost at GPU)
                # greedy algorithm is just as there are only 8 experts for Mixtral
                best_config = -1
                best_cost = float('inf')
                for config in range(1 << len(experts)):
                    sum_cost = 0
                    for i_expert in range(len(experts)):
                        if (config >> i_expert) & 1:
                            sum_cost += cost_per_expert[i_expert, 0]
                        else:
                            sum_cost += cost_per_expert[i_expert, 1]
                    if sum_cost < best_cost:
                        best_cost = sum_cost
                        best_config = config
                
                # then, we can offload the experts according to the best configuration
                cpu_experts = []
                gpu_experts = []
                for i_expert in range(8):
                    if (best_config >> i_expert) & 1:
                        cpu_experts.append(i_expert)
                    else:
                        gpu_experts.append(i_expert)
                
                # TODO: further parallelism is possible
                for i_expert in cpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    current_state = self.run_expert_at_cpu(
                        i_layer, 
                        i_expert, 
                        current_state.to('cpu', non_blocking=True), 
                        routing_weights[top_2_list, idx_list, None].to('cpu', non_blocking=True),
                    )
                    inps_after_experts.index_add_(
                        0, 
                        top_2s[i_expert].to(self.dev, non_blocking=True), 
                        current_state.to(self.dev, non_blocking=True)
                    )
                
                for i_expert in gpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        current_state = experts[i_expert](current_state, routing_weights[top_2_list, idx_list, None])
                    else:
                        self.expert_placeholder.load_state_dict(experts[i_expert].state_dict())
                        current_state = self.expert_placeholder(current_state, routing_weights[top_2_list, idx_list, None])
                    inps_after_experts.index_add_(
                        0, 
                        top_2s[i_expert].to(self.dev, non_blocking=True), 
                        current_state.to(self.dev, non_blocking=True)
                    )

            else: 
                # decode stage with offloading
                assert input_ids.shape[-1] == 1
                expert_0, expert_1 = int(selected_experts[0][0]), int(selected_experts[0][1])
                routing_weights_0, routing_weights_1 = routing_weights[:, 0, None], routing_weights[:, 1, None]

                assert expert_0 != expert_1

                self.cnt_expert_all += 2

                if self.is_expert_in_gpu(i_layer, expert_0):
                    inps_after_experts += experts[expert_0](inps, routing_weights_0)
                    self.cnt_expert_hit += 1
                else:
                    inps_after_experts += self.run_expert_at_cpu(
                        i_layer,
                        expert_0, 
                        inps.to('cpu', non_blocking=True),
                        routing_weights_0.to('cpu', non_blocking=True),
                    ).to(self.dev, non_blocking=True)

                if self.is_expert_in_gpu(i_layer, expert_1):
                    inps_after_experts += experts[expert_1](inps, routing_weights_1)
                    self.cnt_expert_hit += 1
                else:
                    inps_after_experts += self.run_expert_at_cpu(
                        i_layer,
                        expert_1, 
                        inps.to('cpu', non_blocking=True),
                        routing_weights_1.to('cpu', non_blocking=True),
                    ).to(self.dev, non_blocking=True)
            
            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

            # end of one layer
        
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis
    
    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        return self.model.layers[i_layer].block_sparse_moe.experts[i_expert](inps, routing_weights)        

if __name__ == '__main__':
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
    # parser.add_argument(
    #     '--microbench', action='store_true',
    #     help='Run microbenchmark.',
    # )

    args = parser.parse_args() 

    model = FiddlerMixtral(args)
    # if args.microbench:
    #     model.microbench()
    #     exit()
    model.generate(args.input, output_token=args.n_token)
