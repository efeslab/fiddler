# fix numpy in colab
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy
import os
import sys
import argparse
import logging
import pandas as pd

sys.path.append("mixtral-offloading")


def main():
    os.chdir("/home/yilegu/fiddler/benchmarks/mixtral_offloading")

    if args.framework == 'mixtral-offloading':
        logging.info('Using mixtral-offloading')
        model = init_mixtral_offload()
    elif args.framework == 'deepspeed-mii':
        logging.info('Using deepspeed-mii')
        model = init_deepspeed_mii()
    else:
        raise ValueError(f'Unknown framework: {args.framework}')

    eval(model, args.prefill)


def init_deepspeed_mii():
    import deepspeed
    from transformers.deepspeed import HfDeepSpeedConfig

    model_id = "mistralai/Mixtral-8x7B-v0.1"
    ds_config = {
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            }
        },
        "train_micro_batch_size_per_gpu": 1,
    }

    hfdsc = HfDeepSpeedConfig(ds_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16)

    deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    model.eval()

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    return model


def init_mixtral_offload():
    from hqq.core.quantize import BaseQuantizeConfig
    from mixtral_offloading.src.build_model import OffloadConfig, QuantConfig, build_model

    quantized = False

    if not quantized:
        state_path = "Mixtral-8x7B-Instruct-v0.1"
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    else:
        state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"
        model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"

    config = AutoConfig.from_pretrained(model_name)

    device = torch.device("cuda:0")

    ##### Change this to 5 if you have only 12 GB of GPU VRAM #####
    # offload_per_layer = 4
    offload_per_layer = 5
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
        quant_config = QuantConfig(
            ffn_config=ffn_config,
            attn_config=attn_config)
    else:
        quant_config = None

    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )
    return model


def eval(model, prefill=False):
    import random
    import json
    import time

    device = torch.device("cuda:0")

    path_json = '/home/yilegu/fiddler/benchmarks/datasets/ShareGPT_V3_unfiltered_cleaned_split.json'
    with open(path_json, 'r') as f:
        data = json.load(f)
    texts = []
    for d in data:
        if len(d['conversations']) == 0:
            continue
        # the input of the first round
        texts.append(' '.join(d['conversations'][0]['value'].split()))

    logging.info(f'n of input {len(texts)}')
    random.seed(0)
    random.shuffle(texts)

    n_sample = 3
    
    # open a csv file to save the results
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    f = open(f'eval-{timestamp}.csv', 'w')
    f.write('input_token, output_token, batch_size, time, output_token/s\n')
    

    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # input_lengths = [16, 32, 64, 128]
    # output_lengths = [16, 32, 64, 128, 256, 512]
    # input_lengths = [64, 128, 256, 512]
    # input_lengths = [256, 512, 1024, 2048, 4096]
    input_lengths = [32, 64, 128, 256, 512]
    # input_lengths = [8192]
    output_lengths = [64, 128, 256, 512] if not prefill else [1]
    # batch_sizes = [1, 2, 4, 8, 16]
    batch_sizes = [1]
    for input_token in input_lengths:
        for output_token in output_lengths:
            for batch_size in batch_sizes:
    # for input_token in [16, 32, 64, 128]:
    #     for output_token in [16, 32, 64, 128, 256, 512]:
                idx_text = 0
                time_sum = 0
                num_tokens = 0
                logging.info(
                    f'evaluating -- input_token: {input_token}, output_token: {output_token}, batch_size: {batch_size}')
                for _ in range(n_sample):
                    batch = []
                    for _ in range(batch_size):
                        while True:
                            text = texts[idx_text]
                            idx_text += 1
                            if len(text.split()) >= input_token:
                                # enough input length
                                # batch.append(" ".join(text.split()[:input_token]))
                                batch.append(text)
                                break
                    # text = [text, text]
                    # print(f'input text: {text.split()[:input_token]}')
                    input_ids = tokenizer(
                        batch, return_tensors='pt', max_length=input_token, truncation=True)["input_ids"]
                    # input_ids = input_ids[:, :input_token].to(device)
                    input_ids = input_ids.to(device)
                    start_time = time.time()
                    result = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=output_token,
                        min_new_tokens=output_token,
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
                    # print(f'output text: {tokenizer.decode(result["sequences"][0])}')
                    # log input text
                    logging.info(f'input text: {batch}')
                    # decode all the output tokens
                    for i in range(result["sequences"].shape[0]):
                        logging.info(f'{i} output text: {tokenizer.decode(result["sequences"][i])}')
                logging.info(
                    f'*******************\n'
                    f'input_token: {input_token}, output_token: {output_token}, batch_size: {batch_size}\n'
                    f'time: {time_sum / n_sample:.2f}, '
                    f'output token/s: {output_token * batch_size / (time_sum / n_sample):.2f}\n'
                    f'*******************\n')
                f.write(f'{input_token}, {output_token}, {batch_size}, {time_sum / n_sample:.2f}, {output_token * batch_size / (time_sum / n_sample):.2f}\n')
                
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--quantized', type=bool, default=False,
        help='Whether to use quantized model in mixtral-offloading.'
        )
    parser.add_argument(
        '--framework',
        type=str,
        default='mixtral-offloading',
        choices=[
            'mixtral-offloading',
            'deepspeed-mii'],
        help='Which framework to use for evaluation.'
        )
    parser.add_argument(
        '--prefill',
        type=bool,
        default=False,
        help='Whether to test prefill (output token = 1) or decode.'
        )

    args = parser.parse_args()

    # save log to file
    # logging.basicConfig(filename='eval.log', level=logging.INFO)
    # save log to a file with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(filename=f'eval-{timestamp}.log', level=logging.INFO) 
    main()
