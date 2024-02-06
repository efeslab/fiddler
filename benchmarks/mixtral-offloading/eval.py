# fix numpy in colab
import numpy
import sys

sys.path.append("mixtral-offloading")
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.build_model import OffloadConfig, QuantConfig, build_model

### STEP 1 Build Model ###

quantized = False

if quantized == False:
    state_path = "Mixtral-8x7B-Instruct-v0.1"
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
else:
    state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"
    model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"

config = AutoConfig.from_pretrained(model_name)

device = torch.device("cuda:0")

##### Change this to 5 if you have only 12 GB of GPU VRAM #####
# offload_per_layer = 4
offload_per_layer = 7
###############################################################

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256


ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)

if quantized:
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)
else:
    quant_config = None


model = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
)

### STEP 2 Performance Eval ###
import random
import json
import time

path_json = './ShareGPT_V3_unfiltered_cleaned_split.json'
with open(path_json, 'r') as f:
    data = json.load(f)
texts = []
for d in data:
    if len(d['conversations']) == 0:
        continue 
    # the input of the first round
    texts.append(' '.join(d['conversations'][0]['value'].split()))

print('n of input', len(texts))
random.seed(0)
random.shuffle(texts)

n_sample = 10

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

for input_token, output_token in zip([16, 32, 64, 128], [16, 32, 64, 128, 256, 512]):
    idx_text = 0
    time_sum = 0
    num_tokens = 0
    print(f'evaluating -- input_token: {input_token}, output_token: {output_token}')
    for _ in range(n_sample):
        text = texts[idx_text]
        idx_text += 1
        if len(text.split()) < input_token:
            # not enough input length
            continue
        print(f'input text: {text.split()[:input_token]}')
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        start_time = time.time()
        result = model.generate(
            input_ids=input_ids[:, :input_token],
            max_new_tokens=output_token,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        end_time = time.time()
        time_sum += end_time - start_time
        # count the number of tokens in the output
        num_tokens += result["sequences"].shape[1]
        print(f'output text: {tokenizer.decode(result["sequences"][0])}')
        
        
    print(
        f'*******************\n'
        f'input_token: {input_token}, output_token: {output_token}, '
        f'token/s: {num_tokens / time_sum:.2f}'
        f'*******************\n')