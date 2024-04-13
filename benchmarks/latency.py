"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys

sys.path.append("../src")
from fiddler import FiddlerMixtral

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
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference.",
    )
    parser.add_argument("--beam_num", type=int, default=1, help="Beam search number.")

    args = parser.parse_args()

    path_json = "./ShareGPT_V3_unfiltered_cleaned_split.json"
    with open(path_json, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)
    model = FiddlerMixtral(args)
    n_sample = 10

    for input_token in [16, 32, 64, 128]:
        for output_token in [16, 32, 64, 128, 256, 512]:
            idx_text = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            for _ in range(n_sample):
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                prefill_time, decode_time, hit_rate = model.generate(
                    [text], output_token=output_token, input_token=input_token
                )
                prefill_time_sum += prefill_time
                decode_time_sum += decode_time
                hit_rate_sum += hit_rate
            # write to file
            with open("latency.txt", "a") as f:
                f.write(
                    f"input_token: {input_token}, output_token: {output_token}, "
                    f"prefill_time: {prefill_time_sum / n_sample}, "
                    f"decode_time: {decode_time_sum / n_sample}, "
                    f"hit_rate: {hit_rate_sum / n_sample},"
                    f"{output_token *n_sample/ (prefill_time_sum + decode_time_sum):.2f}token/s\n"
                )
