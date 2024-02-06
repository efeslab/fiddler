"""Microbenchmarking for CPU offloading"""
import argparse
import copy
import os
import sys
import time 
import torch 
import json 
import random
import tqdm
import numpy as np

sys.path.append("../src")

from fiddler import FiddlerMixtral

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
        '--input-token', type=int, default=20,
        help='Number of tokens to generate.',
    )
    parser.add_argument(
        '--output-token', type=int, default=20,
        help='Number of tokens to generate.',
    )

    args = parser.parse_args() 

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
    model = FiddlerMixtral(args)
    n_sample = 100

    for input_token, output_token in zip([16, 32, 64, 128], [16, 32, 64, 128, 256, 512]):
        idx_text = 0
        prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
        for _ in range(n_sample):
            text = texts[idx_text]
            idx_text += 1
            if len(text.split()) < input_token:
                # not enough input length
                continue
            prefill_time, decode_time, hit_rate = model.generate(
                text, 
                output_token=output_token, 
                input_token=input_token
            )
            prefill_time_sum += prefill_time
            decode_time_sum += decode_time
            hit_rate_sum += hit_rate
        print(
            f'input_token: {input_token}, output_token: {output_token}, '
            f'prefill_time: {prefill_time_sum / n_sample}, '
            f'decode_time: {decode_time_sum / n_sample}, '
            f'hit_rate: {hit_rate_sum / n_sample}')
