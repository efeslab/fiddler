import copy
import concurrent.futures
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
from bfloat16_expert import cpu_expert
import flashinfer
from attention import SelfAttention


class FiddlerMixtral:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        # kwargs = {"use_flash_attention_2": True}
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            # device_map='cpu',
            use_cache=True,
        )
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        self.vocab_size = self.model.config.vocab_size
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        workspace_buffer = torch.empty(
            16 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cpu_offload = args.cpu_offload

        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
        self.expert_token_num = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.cpu_layer_num = []
        self.outliner_nums = []
        self.outliners = []
        self.cpu_experts = [[] for i in range(self.n_layer)]
        self.beam_width = args.beam_width

        self.torch_threads = args.torch_threads
        self.cpp_threads = args.cpp_threads
        # TODO: find this value based on device config
        self.latency_cpu = 7
        self.latency_gpu = 40

        self.cpu_token_num = 0
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.cpu_expert_time = []
        self.attention_time = []
        self.selection_time = []
        self.gpu_expert_time = []
        self.search_config_time = []
        self.one_token_time = []

        # self.init_attention()

        self.bring_non_expert_to_gpu()

        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)
        # print(self.expert_loc)

        self.bring_expert_to_gpu()

        self.init_cpu_expert()
        self.pin_expert_in_cpu()

        print("Model is ready.")

    def test_cpu_expert(self):
        """Test CPU expert"""
        torch.set_num_threads(self.torch_threads)
        n_sample = 1
        cpp_times = []
        pytorch_times = []
        token_num = 1
        routing_weights = torch.tensor(
            [[1] for i in range(token_num)], dtype=torch.bfloat16
        )

        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if not self.is_expert_in_gpu(i, j):
                    for k in range(n_sample):
                        inp = torch.rand((token_num, 4096), dtype=torch.bfloat16)
                        # print(
                        #     "weight:",
                        #     self.model.layers[i_layer]
                        #     .block_sparse_moe.experts[i_expert]
                        #     .w1.weight[0],
                        # )

                        start_time = time.time()
                        out2 = self.model.layers[i].block_sparse_moe.experts[j](
                            inp, routing_weights
                        )
                        pytorch_times.append(time.time() - start_time)
                        # print(f"CPU1 time: {(time.time()-start_time)*10**6:.2f} us")

                        # print(f"CPU2 time: {(time.time()-start_time)*10**6:.2f} us")
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if not self.is_expert_in_gpu(i, j):
                    for k in range(n_sample):
                        inp = torch.rand((token_num, 4096), dtype=torch.bfloat16)
                        # print(
                        #     "weight:",
                        #     self.model.layers[i_layer]
                        #     .block_sparse_moe.experts[i_expert]
                        #     .w1.weight[0],
                        # )

                        start_time = time.time()
                        out1 = (
                            self.cpu_experts[i][j](inp, self.cpp_threads)
                            * routing_weights
                        )
                        cpp_times.append(time.time() - start_time)
        # compute average time for cpu1_time and cpu2_time
        print(f"Average cpp time: {sum(cpp_times)/len(cpp_times)*10**6:.2f} us")
        print(
            f"Average pytorch time: {sum(pytorch_times)/len(pytorch_times)*10**6:.2f} us"
        )
        # compute varation of cpu1_time and cpu2_time
        print(f"Varation of cpp time: {np.var(cpp_times)*10**6:.2f} us")
        print(f"Varation of pytorch time: {np.var(pytorch_times)*10**6:.2f} us")
        # print(out1)

        # print(out2)
        # delta = torch.abs(out1 - out2)
        # print(f"Max delta: {delta.max()}")

    def init_attention(self):
        """Initialize attention"""
        for i in range(self.n_layer):
            self.model.layers[i].self_attn = SelfAttention(
                self.model.config, self.model.layers[i].self_attn
            )

    def init_cpu_expert(self):
        """Initialize CPU expert"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                expert = cpu_expert(
                    self.model.layers[i].block_sparse_moe.experts[j].w1.weight,
                    self.model.layers[i].block_sparse_moe.experts[j].w2.weight,
                    self.model.layers[i].block_sparse_moe.experts[j].w3.weight,
                )
                self.cpu_experts[i].append(expert)

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

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """Set the location of experts"""
        if popular_experts is None:
            # list of (i_layer, i_expert) in the order of popularity
            # determined based on profile
            popular_experts = [
                (9, 5),
                (11, 2),
                (10, 4),
                (28, 0),
                (13, 1),
                (17, 7),
                (12, 1),
                (8, 6),
                (16, 1),
                (9, 0),
                (14, 5),
                (19, 5),
                (26, 2),
                (30, 7),
                (7, 1),
                (3, 7),
                (23, 4),
                (22, 1),
                (29, 3),
                (1, 5),
                (13, 0),
                (5, 1),
                (18, 0),
                (4, 7),
                (10, 3),
                (1, 2),
                (3, 0),
                (8, 3),
                (11, 0),
                (11, 5),
                (11, 1),
                (31, 4),
                (21, 0),
                (25, 1),
                (15, 5),
                (22, 4),
                (27, 5),
                (16, 7),
                (15, 1),
                (13, 2),
                (15, 4),
                (21, 1),
                (27, 7),
                (9, 7),
                (7, 4),
                (31, 5),
                (2, 1),
                (11, 6),
                (12, 3),
                (2, 4),
                (24, 2),
                (28, 2),
                (0, 2),
                (30, 2),
                (6, 0),
                (6, 7),
                (15, 6),
                (6, 2),
                (14, 2),
                (2, 0),
                (17, 2),
                (19, 2),
                (24, 0),
                (10, 0),
                (19, 4),
                (1, 4),
                (26, 3),
                (31, 7),
                (17, 6),
                (25, 3),
                (12, 6),
                (0, 0),
                (26, 0),
                (29, 7),
                (27, 2),
                (19, 6),
                (5, 0),
                (18, 2),
                (20, 1),
                (12, 4),
                (17, 5),
                (5, 4),
                (30, 6),
                (20, 5),
                (24, 6),
                (25, 2),
                (28, 4),
                (4, 6),
                (7, 2),
                (20, 3),
                (23, 2),
                (8, 4),
                (30, 0),
                (3, 4),
                (12, 5),
                (23, 7),
                (1, 7),
                (22, 5),
                (18, 4),
                (31, 0),
                (17, 0),
                (0, 5),
                (14, 6),
                (0, 3),
                (15, 7),
                (5, 6),
                (4, 4),
                (24, 7),
                (31, 1),
                (27, 6),
                (22, 2),
                (14, 1),
                (1, 0),
                (29, 1),
                (21, 3),
                (25, 7),
                (22, 3),
                (7, 3),
                (2, 6),
                (29, 5),
                (28, 3),
                (6, 6),
                (7, 5),
                (5, 7),
                (8, 5),
                (20, 4),
                (21, 5),
                (18, 7),
                (27, 0),
                (16, 0),
                (24, 5),
                (12, 2),
                (2, 2),
                (24, 3),
                (4, 1),
                (29, 0),
                (3, 1),
                (21, 6),
                (10, 2),
                (20, 7),
                (19, 0),
                (26, 7),
                (20, 6),
                (23, 3),
                (4, 3),
                (30, 1),
                (1, 6),
                (29, 2),
                (30, 3),
                (0, 6),
                (8, 1),
                (25, 6),
                (29, 4),
                (16, 2),
                (23, 1),
                (26, 1),
                (26, 6),
                (16, 4),
                (2, 5),
                (0, 4),
                (7, 6),
                (14, 4),
                (3, 6),
                (20, 0),
                (18, 3),
                (4, 5),
                (17, 4),
                (0, 1),
                (16, 5),
                (19, 3),
                (23, 0),
                (30, 4),
                (20, 2),
                (13, 6),
                (18, 6),
                (15, 2),
                (3, 5),
                (22, 0),
                (10, 1),
                (9, 6),
                (10, 5),
                (25, 4),
                (9, 2),
                (18, 1),
                (6, 4),
                (4, 2),
                (23, 5),
                (6, 5),
                (21, 2),
                (5, 5),
                (6, 1),
                (26, 5),
                (12, 0),
                (25, 0),
                (4, 0),
                (14, 0),
                (16, 6),
                (31, 2),
                (8, 0),
                (21, 7),
                (14, 3),
                (31, 6),
                (28, 1),
                (5, 3),
                (23, 6),
                (6, 3),
                (18, 5),
                (25, 5),
                (27, 1),
                (11, 7),
                (11, 4),
                (24, 1),
                (0, 7),
                (8, 7),
                (13, 3),
                (21, 4),
                (27, 4),
                (13, 7),
                (3, 2),
                (9, 1),
                (2, 7),
                (7, 0),
                (2, 3),
                (28, 5),
                (27, 3),
                (15, 0),
                (24, 4),
                (5, 2),
                (22, 6),
                (3, 3),
                (28, 6),
                (14, 7),
                (13, 4),
                (28, 7),
                (22, 7),
                (13, 5),
                (19, 1),
                (26, 4),
                (1, 1),
                (17, 1),
                (16, 3),
                (10, 7),
                (29, 6),
                (19, 7),
                (31, 3),
                (7, 7),
                (1, 3),
                (8, 2),
                (9, 4),
                (17, 3),
                (30, 5),
                (15, 3),
                (9, 3),
                (10, 6),
                (12, 7),
                (11, 3),
            ]

        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)

    def pin_expert_in_cpu(self):
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if not self.is_expert_in_gpu(i, j):
                    for name in ["w1", "w2", "w3"]:
                        w = getattr(
                            self.model.layers[i].block_sparse_moe.experts[j], name
                        )
                        src_weight_data_tensor = w.weight.data
                        pinned = src_weight_data_tensor.pin_memory()
                        w.weight.data = pinned

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1

    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev)
        return int((free_mem) // (n_param * 2))

    def initial_beam_tensor(self, input_tensor):
        # transfer tensor of shape (batch_size*beam_width, seq_len, beam_width) to (batch_size*beam_width, 1) properly
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [
                i * self.beam_width
                for i in range(input_tensor.shape[0] // self.beam_width)
            ]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(
        self,
        texts=None,
        output_token=20,
        input_token=None,
        input_ids=None,
        beam_width=1,
    ):
        torch.set_num_threads(self.torch_threads)
        self.past_key_values_length = 0
        self.cpu_token_num = 0
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.cpu_expert_time = []
        self.gpu_expert_time = []
        self.selection_time = []
        self.attention_time = []
        self.search_config_time = []
        self.one_token_time = []
        self.outliner_nums = []
        self.cpu_layer_num = []
        self.outliners = []
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        # input_ids.shape: (batch_size, seq_len)
        # position_ids.shape: (1,seq_len)
        self.beam_width = beam_width
        input_ids, position_ids = self.tokenize(texts)

        # self.page_size = 16
        # self.max_num_pages = self.beam_width * (
        #     (input_ids.shape[-1] + output_token) // self.page_size + 1
        # )
        # self.num_kv_heads = self.model.config.num_key_value_heads
        # self.num_qo_heads = self.model.config.num_attention_heads
        # self.head_dim = self.model.config.hidden_size // self.num_qo_heads
        # self.kv_data_at_layer = [
        #     torch.randn(
        #         self.max_num_pages,
        #         2,
        #         self.page_size,
        #         self.num_kv_heads,
        #         self.head_dim,
        #         dtype=torch.float16,
        #         device="cuda:0",
        #     )
        #     for _ in range(self.n_layer)
        # ]
        # self.kv_indptr = torch.tensor(
        #     [i for i in range(self.beam_width + 1)],
        #     dtype=torch.int32,
        #     device="cuda:0",
        # )
        # self.kv_indices = torch.tensor(
        #     [i for i in range(self.beam_width)],
        #     dtype=torch.int32,
        #     device="cuda:0",
        # )
        # self.kv_last_page_len = torch.tensor(
        #     [0 for i in range(self.beam_width)], dtype=torch.int32, device="cuda:0"
        # )
        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)
        tick = time.time()
        for i_token in range(output_token):
            start_time = time.time()
            if is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])
                    # print("--------------------")
                    # print(f"beam[{i}]: {decode_strings[i]}")

            # self.qo_indptr = torch.tensor(
            #     [i * input_ids.shape[-1] for i in range(self.beam_width + 1)],
            #     dtype=torch.int32,
            #     device="cuda:0",
            # )

            logits = self.mixtral_forward(
                input_ids,
                position_ids,
                is_decode,
            )

            logits = logits.to("cpu")
            # logits.shape: (batch_size, seq_len, vocab_size)

            # normalize logits
            logits = F.softmax(logits, dim=-1)
            self.past_key_values_length += logits.shape[1]
            # greedy search:
            if self.beam_width == 1:
                output = torch.argmax(logits, dim=-1)
                input_ids = output[:, -1].unsqueeze(0).view(-1, 1).to(self.dev)
            else:
                # beam_search:

                if search_start:
                    new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                    new_probs = new_probs[:, -1].flatten().view(-1, 1)
                    for i in range(probs.shape[0]):
                        for j in range(probs.shape[0]):
                            new_probs[i * self.beam_width + j] *= probs[i]
                    topk_probs, topk_idx = torch.topk(new_probs, self.beam_width, dim=0)
                    probs = topk_probs
                    output = output.flatten().view(-1, 1)[
                        topk_idx.view(-1, self.beam_width).flatten()
                    ]
                    # print(output)
                    # exit(0)
                    input_ids = output.to(self.dev)
                else:
                    new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                    new_probs = self.initial_beam_tensor(new_probs)
                    output = self.initial_beam_tensor(output)
                    search_start = True
                    probs = probs * new_probs
                    input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
                # new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            # input_ids.shape: (batch_size, seq_len=1)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .view(-1, 1)
            )
            # position_ids.shape: (1, 1)
            if not is_decode:
                torch.cuda.synchronize()
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
            self.one_token_time.append((time.time() - start_time) * 10**3)
        torch.cuda.synchronize()
        decode_time = time.time() - tick
        # if self.beam_width == 1:
        #     for i in range(input_ids.shape[0]):
        # print("--------------------")
        # print(f"Input: {texts[i]}")
        # print(f"Output: {decode_strings[i]}")
        # else:
        #     probs = probs.view(-1, self.beam_width)
        #     max_ids = torch.argmax(probs, dim=-1)
        # for i in range(max_ids.shape[0]):
        # print("--------------------")
        # print(f"Input: {texts[i]}")
        # print(f"Output: {decode_strings[i * self.beam_width + max_ids[i]]}")

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
        )

    def tokenize(self, texts):
        input_ids = []
        for text in texts:
            encodings = self.tokenizer(text, return_tensors="pt")
            input_id = encodings.input_ids.to(self.dev)
            for i in range(self.beam_width):
                input_ids.append(input_id[0])

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.dev)

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(
        self,
        input_ids,
        position_ids,
        is_decode,
    ):
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps)
        # cpu_layer_num = 0
        # outliner_num = 0
        # outliners = []

        # self.prefill_wrapper.begin_forward(
        #     self.qo_indptr,
        #     self.kv_indptr,
        #     self.kv_indices,
        #     self.kv_last_page_len,
        #     self.num_qo_heads,
        #     self.num_kv_heads,
        #     self.head_dim,
        # )
        # print("--------------------")
        # print(self.qo_indptr, self.kv_indptr, self.kv_indices, self.kv_last_page_len)

        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape
            inps_residual = inps
            # start_time = time.time()
            inps = layer.input_layernorm(inps)
            # (
            #     inps,
            #     new_paged_kv_data,
            #     new_kv_indptr,
            #     new_kv_indices,
            #     new_kv_last_page_len,
            # ) = layer.self_attn.forward(
            #     inps,
            #     self.kv_data_at_layer[i_layer],
            #     self.qo_indptr,
            #     self.prefill_wrapper,
            # )
            # self.kv_data_at_layer[i_layer] = new_paged_kv_data

            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )

            # exit(0)
            # inps.shape: (batch_size, seq_len/token_num, embed_dim)
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            # torch.cuda.synchronize()
            # self.attention_time.append((time.time() - start_time) * 10**6)
            inps = inps.view(-1, hidden_dim)
            # start_time = time.time()
            # print(f"Attention time:{(time.time()-start_time)*10**3}")
            # inps.shape: (batch_size*seq_len*embed_dim/hidden_dim, hidden_dim)
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            # routing_weights.shape: (batch_size*seq_len, num_experts)
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            # routing_weights.shape: (batch_size*seq_len, 2)
            # selected_experts.shape: (batch_size*seq_len, 2)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # print(f"Selection time:{(time.time()-start_time)*10**3}")
            # torch.cuda.synchronize()
            # self.selection_time.append((time.time() - start_time) * 10**6)
            # for top_2 in selected_experts:
            #     for i in top_2:
            #         self.expert_token_num[i_layer][i] += 1

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            if self.cpu_offload == 0:
                # baseline: do everything at GPU
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

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
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to("cpu")

                    # end of one expert

            else:
                # prefill stage with offloading
                # start_time = time.time()
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                # first, calculate the number of tokens for each expert
                idxs, top_2s = [], []
                # cost_per_expert = np.zeros(
                #     (len(experts), 2), dtype=float
                # )  # 0: CPU, 1: GPU
                # hit_cnt = self.cnt_expert_hit
                cpu_experts = []
                gpu_experts = []
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])
                    idxs.append(idx)
                    top_2s.append(top_2)
                    # expected latency at CPU: number of token * cost_at_cpu
                    # expected latency at GPU: cost_at_gpu (constant)
                    cpu_cost = top_2.shape[0] * self.latency_cpu
                    gpu_cost = self.latency_gpu + cpu_cost * 0.1
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # if the expert is in GPU, the latency at GPU is
                        # approximately 0
                        gpu_cost = cpu_cost * 0.1
                        self.cnt_expert_hit += top_2.shape[0]
                    self.cnt_expert_all += top_2.shape[0]
                    if cpu_cost <= gpu_cost:
                        cpu_experts.append(i_expert)
                    else:
                        gpu_experts.append(i_expert)
                # print("hit number of this layer:", self.cnt_expert_hit - hit_cnt)
                # print("Number of tokens for each expert:", expert_tokens)

                # second, partition experts processing between CPU and GPU so that we can minimize:
                # max(sum of cost at CPU, sum of cost at GPU)
                # greedy algorithm is just as there are only 8 experts for
                # Mixtral
                # print(cpu_experts, gpu_experts)
                # torch.cuda.synchronize()
                # self.search_config_time.append((time.time() - start_time) * 10**6)
                # start_time = time.time()
                # use_gpu = False
                # gpu_token_num = 0
                for i_expert in gpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    if len(top_2_list) == 0:
                        continue
                    # use_gpu = True
                    # gpu_token_num += len(top_2_list)
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        # self.expert_placeholder.load_state_dict(
                        #     experts[i_expert].state_dict()
                        # )
                        for name in ["w1", "w2", "w3"]:
                            dst = getattr(self.expert_placeholder, name).weight.data
                            src = getattr(experts[i_expert], name).weight.data
                            dst.copy_(src)
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )
                # torch.cuda.synchronize()
                # if use_gpu:
                #     self.gpu_expert_time.append(
                #         (time.time() - start_time) * 10**6 / gpu_token_num
                #     )

                # use_cpu = False
                # cpu_start = time.time()
                # cpu_token_num = 0
                for i_expert in cpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    if len(top_2_list) == 0:
                        continue
                    # use_cpu = True
                    # cpu_token_num += len(top_2_list)
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    current_state = self.run_expert_at_cpu(
                        i_layer,
                        i_expert,
                        current_state.to("cpu"),
                        routing_weights[top_2_list, idx_list, None].to("cpu"),
                    )
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )
                # lock = threading.Lock()
                # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                #     futures = dict()
                #     for i_expert in cpu_experts:
                #         top_2_list = top_2s[i_expert].tolist()
                #         if len(top_2_list) == 0:
                #             continue
                #         use_cpu = True
                #         cpu_token_num += len(top_2_list)
                #         idx_list = idxs[i_expert].tolist()
                #         current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                #         futures.update(
                #             {
                #                 executor.submit(
                #                     self.run_expert_at_cpu,
                #                     i_layer,
                #                     i_expert,
                #                     current_state.to("cpu"),
                #                     routing_weights[top_2_list, idx_list, None].to(
                #                         "cpu"
                #                     ),
                #                 ): i_expert
                #             }
                #         )
                #         # # measure data transfer time
                #         # current_state = self.run_expert_at_cpu(
                #         #     i_layer,
                #         #     i_expert,
                #         #     current_state.to("cpu"),
                #         #     routing_weights[top_2_list, idx_list, None].to("cpu"),
                #         # )
                #     for future in concurrent.futures.as_completed(futures):
                #         i_expert = futures[future]
                #         current_state = future.result()
                #         with lock:
                #             inps_after_experts.index_add_(
                #                 0,
                #                 top_2s[i_expert].to(self.dev, non_blocking=True),
                #                 current_state.to(self.dev, non_blocking=True),
                #             )
                # torch.cuda.synchronize()
                # if use_cpu:
                #     cpu_time = (time.time() - cpu_start) * 10**6 / cpu_token_num
                #     if cpu_time > 10000:
                #         # print(
                #         #     f"Layer {i_layer} CPU time: {cpu_time:.2f} us, token num: {cpu_token_num}"
                #         # )
                #         # print(f"CPU Experts: {cpu_experts}, inps shape: {inps.shape}")
                #         outliner_num += 1
                #         outliners.append(cpu_time * cpu_token_num)
                #         # exit(0)
                #     self.cpu_expert_time.append(cpu_time)
                # if use_cpu:
                #     cpu_layer_num += 1
                # expert_time = time.time() - start_time
                # print(f"Expert time: {expert_time}")
            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)
            # layer_time = time.time() - layer_start
            # print(f"Layer time: {layer_time}")

            # end of one layer

        # self.cpu_layer_num.append(cpu_layer_num)
        # self.outliner_nums.append(outliner_num)
        # self.outliners.extend(outliners)
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)
        # self.prefill_wrapper.end_forward()
        self.past_key_value = present_key_value
        # print("//////////////////////")
        # print(new_kv_indptr, new_kv_indices, new_kv_last_page_len)
        # self.kv_indptr = new_kv_indptr
        # self.kv_indices = new_kv_indices
        # self.kv_last_page_len = new_kv_last_page_len

        return lm_logis

    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        # return self.model.layers[i_layer].block_sparse_moe.experts[i_expert](
        #     inps, routing_weights
        # )
        return (
            self.cpu_experts[i_layer][i_expert](inps, self.cpp_threads)
            * routing_weights
        )

    def write_expert_hit_num(self, filename):
        with open(filename, "w") as f:
            for i_layer, expert_hit_num in enumerate(self.expert_token_num):
                f.write(f"Layer {i_layer}\n")
                for hit_num in expert_hit_num:
                    f.write(f"{hit_num},")
                f.write("\n")

    def write_popular_experts(self, filename):
        popular_experts = []
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                popular_experts.append((i, j, self.expert_token_num[i][j]))
        popular_experts.sort(key=lambda x: x[2], reverse=True)
        with open(filename, "w") as f:
            for i, j, hit_num in popular_experts:
                f.write(f"{i*self.n_expert+j},{hit_num}\n")

    def reset_popular_experts(self):
        self.expert_token_num = np.zeros((self.n_layer, self.n_expert), dtype=int)
