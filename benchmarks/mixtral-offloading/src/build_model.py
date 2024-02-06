import os
import json
from functools import cache
from dataclasses import dataclass
import typing as tp

import torch
from torch import nn

from transformers import AutoConfig
from transformers.models.mixtral import MixtralForCausalLM, MixtralConfig

from safetensors.torch import load_file
from safetensors import safe_open

from torch import nn
from tqdm.auto import trange

from hqq.core.quantize import BaseQuantizeConfig

from .expert_cache import ExpertCache
from .expert_wrapper import MixtralExpertWrapper
from .custom_layers import (
    HQQLinearTritonSavable,
    MixtralBlockSparseTop2MLP_HQQ,
    MixtralBlockSparseTop2MLP,
    SparseMoEWrapper,
)
from .utils import with_default_dtype


@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int


class QuantConfig:
    def __init__(
        self,
        ffn_config: BaseQuantizeConfig,
        attn_config: BaseQuantizeConfig,
    ):
        self.ffn_config = ffn_config
        self.attn_config = attn_config

    @cache
    def get_ffn_metas(self, hidden_dim: int, ffn_dim: int) -> tuple[tp.Any, tp.Any]:
        return (
            HQQLinearTritonSavable.get_hqq_meta((hidden_dim, ffn_dim), self.ffn_config),
            HQQLinearTritonSavable.get_hqq_meta((ffn_dim, hidden_dim), self.ffn_config),
        )


def replace_attn_layers(
    model: MixtralForCausalLM,
    config: MixtralConfig,
    quant_config: QuantConfig | None,
    device: torch.device,
) -> None:
    
    for layer in model.model.layers:
        layer.block_sparse_moe.gate = nn.Linear(
            config.hidden_size,
            config.num_local_experts,
            dtype=torch.float16,
            device=device,
            bias=False,
        )

    if quant_config is None:
        return
    
    attn_quant_config = quant_config.attn_config

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads

    shapes = [
        (hidden_size, num_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (num_heads * head_dim, hidden_size),
    ]

    shape_to_meta = {
        shape: HQQLinearTritonSavable.get_hqq_meta(shape, attn_quant_config)
        for shape in shapes
    }

    def patch_fct_hqq(shape, quant_config):
        meta = shape_to_meta[shape]
        layer = HQQLinearTritonSavable(None, quant_config, meta=meta)
        return layer

    for layer in model.model.layers:

        layer.self_attn.q_proj = patch_fct_hqq(
            (hidden_size, num_heads * head_dim), attn_quant_config
        )
        layer.self_attn.k_proj = patch_fct_hqq(
            (hidden_size, num_key_value_heads * head_dim), attn_quant_config
        )
        layer.self_attn.v_proj = patch_fct_hqq(
            (hidden_size, num_key_value_heads * head_dim), attn_quant_config
        )
        layer.self_attn.o_proj = patch_fct_hqq(
            (hidden_size, num_heads * head_dim), attn_quant_config
        )


@cache
def get_default_ffn_quant_config(ffn_dim: int = 14336, hidden_dim: int = 4096):
    quant_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )

    meta1 = HQQLinearTritonSavable.get_hqq_meta((hidden_dim, ffn_dim), quant_config)
    meta2 = HQQLinearTritonSavable.get_hqq_meta((ffn_dim, hidden_dim), quant_config)

    return quant_config, meta1, meta2


def make_empty_expert(
    model_config: MixtralConfig, quant_config: QuantConfig
) -> MixtralBlockSparseTop2MLP_HQQ | MixtralBlockSparseTop2MLP:
    if quant_config is None:
        return MixtralBlockSparseTop2MLP(
            model_config
        )

    meta1, meta2 = quant_config.get_ffn_metas(
        model_config.hidden_size, model_config.intermediate_size
    )
    return MixtralBlockSparseTop2MLP_HQQ(
        model_config,
        quant_config.ffn_config,
        meta1,
        meta2,
    )


def make_and_load_expert_wrapper(
    config: MixtralConfig,
    quant_config: QuantConfig | None,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> MixtralExpertWrapper:
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        json_config = json.load(f)
        if quant_config is None:
            state_dict = {}
            for i in range(1, 4):
                module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w{i}.weight"
                state_fpath = json_config["weight_map"][module_idx]
                # we need only the expert 0 part from consolidated weight
                with safe_open(os.path.join(states_dir, state_fpath), framework="pt") as f_weight:
                    base = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
                    state_dict[f"w{i}.weight"] = f_weight.get_tensor(f"{base}.w{i}.weight")
            expert = make_empty_expert(config, quant_config)
            expert.load_state_dict(state_dict, strict=True)
        else:
            module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
            state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.W_q"]
            state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
            expert = make_empty_expert(config, quant_config)
            expert.load_state_dict(state_dict, strict=True)

    quantized = True if quant_config is not None else False
    return MixtralExpertWrapper(expert, device, quantized)


def load_00_expert_state_dict(states_dir: str, device: torch.device, quant_config: QuantConfig | None):
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        if quant_config is None:
            module_idx = f"model.layers.0.block_sparse_moe.experts.0.w1.weight"
            state_fpath = json.load(f)["weight_map"][module_idx]
            with safe_open(os.path.join(states_dir, state_fpath), framework="pt") as f:
                state_dict = {}
                for i in range(1, 4):
                    state_dict[f"w{i}.weight"] = f.get_tensor(f"model.layers.0.block_sparse_moe.experts.0.w{i}.weight")
                return state_dict
        else:
            module_idx = f"model.layers.0.block_sparse_moe.experts.0"
            state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.W_q"]
            return load_file(os.path.join(states_dir, state_fpath), device=str(device))


def build_model(
    device: torch.device,
    quant_config: QuantConfig | None,
    offload_config: OffloadConfig,
    state_path: str,
):
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    state_dict_00 = load_00_expert_state_dict(state_path, device, quant_config)

    def _make_module():
        config = AutoConfig.from_pretrained(model_name)
        expert = make_empty_expert(config, quant_config)
        expert.load_state_dict(state_dict_00, strict=False)
        quantized = True if quant_config is not None else False
        return MixtralExpertWrapper(expert, device=device, quantized=quantized)

    with device, with_default_dtype(torch.float16):
        model = MixtralForCausalLM(
            AutoConfig.from_pretrained(
                model_name,
                num_local_experts=0,
                torch_dtype=torch.float16,
                device_map=device,
            ),
        )

    model_config = AutoConfig.from_pretrained(model_name)
    
    replace_attn_layers(model, model_config, quant_config, device)
    
    if quant_config is not None:
        state_index_path = os.path.join(state_path, "model.safetensors.index.json")
        with open(state_index_path) as f:
            weight_map = json.load(f)["weight_map"]

        trunk_state_path = os.path.join(
            state_path,
            weight_map["model.embed_tokens.weight"],
        )
        # this is possible because all non-expert weights are in this one file
        model.load_state_dict(load_file(trunk_state_path, device=str(device)), strict=False)
    else:
        # glob all safetensors files and load them
        for root, dirs, files in os.walk(state_path):
            state_dict = {}
            for file in files:
                if file.endswith(".safetensors"):
                    state_dict.update(load_file(os.path.join(root, file)))
            # print(f"state_dict keys: {sorted(state_dict.keys())}")
            print("state_dict keys")
            for key in sorted(state_dict.keys()):
                print(key)
            model.load_state_dict(state_dict, strict=False)
    

    expert_cache = ExpertCache(
        make_module=_make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
    )
    for layer_idx in trange(model_config.num_hidden_layers, desc="Loading experts"):
        curr_layer = model.model.layers[layer_idx]
        curr_layer.block_sparse_moe = SparseMoEWrapper(
            model_config,
            layer_idx,
            curr_layer.block_sparse_moe.gate,
            expert_cache,
        )

        for expert_idx in range(model_config.num_local_experts):
            do_offload = expert_idx < offload_config.offload_per_layer

            # we only need to create experts on cpu storage, whether they will be on gpu is decided by the expert_cache
            expert_wrapper = make_and_load_expert_wrapper(
                config=model_config,
                quant_config=quant_config,
                states_dir=state_path,
                expert_uid=(layer_idx, expert_idx),
                device=torch.device("cpu"),
            )

            expert_cache.add_expert(
                uid=(layer_idx, expert_idx),
                module=expert_wrapper,
                eviction_group=layer_idx,
                offload=do_offload,
            )

            del expert_wrapper
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    return model