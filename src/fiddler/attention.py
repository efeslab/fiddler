import copy
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flashinfer


class SelfAttention(torch.nn.Module):
    def __init__(self, config, self_attn):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def update_kv_cache(
        self,
        paged_kv_data,
        key_states,
        value_states,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
    ):
        max_num_pages, _, page_size, _, _ = paged_kv_data.size()
        empty_page_indices = torch.arange(0, max_num_pages, dtype=torch.int32)
        mask = torch.ones(max_num_pages, dtype=torch.bool)
        mask[paged_kv_indices] = False
        available_page_indices = empty_page_indices[mask].to("cuda:0")
        for i in range(len(qo_indptr) - 1):
            last_page_len = paged_kv_last_page_len[i].item()
            seq_len = (qo_indptr[i + 1] - qo_indptr[i]).item()
            last_page_index = paged_kv_indices[paged_kv_indptr[i + 1] - 1].item()
            if seq_len + last_page_len > page_size:
                page_num, new_last_page_len = divmod(
                    seq_len + last_page_len - page_size,
                    page_size,
                )
                if len(available_page_indices) < page_num + 1:
                    raise ValueError("No available page to cache")
                page_index = available_page_indices[0:page_num]
                available_page_indices = available_page_indices[page_num:]
                new_last_page_index = available_page_indices[0]
                available_page_indices = available_page_indices[1:]
                for j in range(page_size - last_page_len):
                    paged_kv_data[last_page_index, 0, last_page_len + j] = key_states[
                        qo_indptr[i] + j
                    ]
                    paged_kv_data[last_page_index, 1, last_page_len + j] = value_states[
                        qo_indptr[i] + j
                    ]
                for j, idx in enumerate(page_index):
                    left = (
                        qo_indptr[i].item() + page_size - last_page_len + j * page_size
                    )
                    right = (
                        qo_indptr[i].item()
                        + page_size
                        - last_page_len
                        + (j + 1) * page_size
                    )

                    paged_kv_data[idx, 0, :] = key_states[left:right, :]
                    paged_kv_data[idx, 1, :] = value_states[left:right, :]
                base = (
                    page_num * page_size
                    + qo_indptr[i].item()
                    - last_page_len
                    + page_size
                )
                for j in range(new_last_page_len):
                    paged_kv_data[new_last_page_index, 0, j] = key_states[base + j]
                    paged_kv_data[new_last_page_index, 1, j] = value_states[base + j]

                paged_kv_indices = torch.cat(
                    (
                        torch.cat(
                            (
                                paged_kv_indices[: paged_kv_indptr[i + 1]],
                                page_index,
                                new_last_page_index.view(1),
                            ),
                            dim=0,
                        ),
                        paged_kv_indices[paged_kv_indptr[i + 1] :],
                    ),
                    dim=0,
                )
                for j in range(i + 1, len(paged_kv_indptr)):
                    paged_kv_indptr[j] += page_num + 1
                paged_kv_last_page_len[i] = new_last_page_len
            else:
                paged_kv_last_page_len[i] += seq_len
                for j in range(seq_len):
                    paged_kv_data[last_page_index, 0, last_page_len + j] = key_states[
                        qo_indptr[i] + j
                    ]
                    paged_kv_data[last_page_index, 1, last_page_len + j] = value_states[
                        qo_indptr[i] + j
                    ]
        return paged_kv_data, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        paged_kv_data: torch.Tensor,
        qo_indptr: torch.Tensor,
        prefill_wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    ):
        # paged_kv_data: [max_num_pages, 2, page_size, num_kv_heads, head_dim]
        bsz, q_len, hidden_size = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(-1, self.num_key_value_heads, self.head_dim)

        new_kv_indptr = prefill_wrapper._paged_kv_indptr.clone()
        new_kv_indices = prefill_wrapper._paged_kv_indices.clone()
        new_kv_last_page_len = prefill_wrapper._paged_kv_last_page_len.clone()

        paged_kv_data, new_kv_indptr, new_kv_indices, new_kv_last_page_len = (
            self.update_kv_cache(
                paged_kv_data,
                key_states,
                value_states,
                qo_indptr,
                new_kv_indptr,
                new_kv_indices,
                new_kv_last_page_len,
            )
        )
        output = prefill_wrapper.forward(
            query_states,
            paged_kv_data,
            causal=self.is_causal,
            pos_encoding_mode="ROPE_LLAMA",
            rope_theta=self.rope_theta,
        )
        output = output.reshape(bsz, q_len, hidden_size)
        output = self.o_proj(output)

        return (
            output,
            paged_kv_data,
            new_kv_indptr,
            new_kv_indices,
            new_kv_last_page_len,
        )
