"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
import numpy as np

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

    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width.")
    parser.add_argument("--torch_threads", type=int, default=16, help="Torch threads.")
    parser.add_argument("--cpp_threads", type=int, default=44, help="C++ threads.")

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
    n_sample = 3
    for input_token in [16]:
        for output_token in [64]:
            idx_text = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            print(f"input_token: {input_token}, output_token: {output_token}")
            for _ in range(1):
                idx_text = 0
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                # print("text:", text)
                prefill_time, decode_time, hit_rate = model.generate(
                    [text], output_token=output_token, input_token=input_token
                )
    # for input_token in [16, 32, 64, 128]:
    #     for output_token in [16, 32, 64, 128, 256, 512]:
    for input_token in [32]:
        for output_token in [128, 256, 512]:
            idx_text = 0
            while True:
                text = texts[idx_text]
                idx_text += 1
                if len(text.split()) >= input_token:
                    # enough input length
                    break
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            print(f"input_token: {input_token}, output_token: {output_token}")
            for _ in range(n_sample):

                # print("text:", text)
                prefill_time, decode_time, hit_rate = model.generate(
                    [text], output_token=output_token, input_token=input_token
                )
                print(
                    "prefill_time:",
                    prefill_time,
                    "decode_time:",
                    decode_time,
                )
                print(max(model.cpu_expert_time), min(model.cpu_expert_time))
                # print(model.outliner_nums)
                print(sum(model.outliner_nums), len(model.outliners))
                # print(sum(model.outliners) / len(model.outliners))
                print(
                    "Est improved performance:",
                    prefill_time + decode_time - sum(model.outliners) * 0.88 / 10**6,
                )
                print(
                    f"CPU Layer Num: | {sum(model.cpu_layer_num)/len(model.cpu_layer_num):.2f} | {np.var(model.cpu_layer_num):.2f}"
                )
                print(
                    f"OneToken | {sum(model.one_token_time)/len(model.one_token_time):.2f} ms | {np.var(model.one_token_time):.2f} ms"
                )
                print("         | Average value | Variation | Portion")
                print(
                    f"CPUExpert | {sum(model.cpu_expert_time)/len(model.cpu_expert_time):.2f} | {np.var(model.cpu_expert_time):.2f} | {sum(model.cpu_expert_time)/(decode_time+prefill_time)/10**6:.2f}"
                )
                print(
                    f"GPUExpert | {sum(model.gpu_expert_time)/len(model.gpu_expert_time):.2f} | {np.var(model.gpu_expert_time):.2f} | {sum(model.gpu_expert_time)/(decode_time+prefill_time)/10**6:.2f}"
                )
                print(
                    f"Attention | {sum(model.attention_time)/len(model.attention_time):.2f} | {np.var(model.attention_time):.2f} | {sum(model.attention_time)/(decode_time+prefill_time)/10**6:.2f}"
                )
                print(
                    f"Selection | {sum(model.selection_time)/len(model.selection_time):.2f} | {np.var(model.selection_time):.2f} | {sum(model.selection_time)/(decode_time+prefill_time)/10**6:.2f}"
                )
                print(
                    f"Optconfig | {sum(model.search_config_time)/len(model.search_config_time):.2f} | {np.var(model.search_config_time):.2f} | {sum(model.search_config_time)/(decode_time+prefill_time)/10**6:.2f}"
                )

                prefill_time_sum += prefill_time
                decode_time_sum += decode_time
                hit_rate_sum += hit_rate
            # write to file
            with open(
                f"./results/latency-{args.torch_threads}-{args.cpp_threads}.txt", "a"
            ) as f:
                f.write(
                    f"input_token: {input_token}, output_token: {output_token}, "
                    f"prefill_time: {prefill_time_sum / n_sample}, "
                    f"decode_time: {decode_time_sum / n_sample}, "
                    f"cpu_token_num: {model.cpu_token_num},"
                    f"{output_token *n_sample/ (decode_time_sum+prefill_time_sum):.2f}token/s\n"
                )
