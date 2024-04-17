import argparse
import os

# import utils
import random
from mixtral import FiddlerMixtral
import torch
import time
import datasets


# def test_generate(args, text):
#     model = FiddlerMixtral(args)
#     prefill_times = []
#     decode_times = []
#     hit_rates = []
#     # batch_sizes = list(range(1, args.batch_size + 1))
#     batch_sizes = [2**i for i in range(0, int(args.batch_size.bit_length()))]
#     for i in batch_sizes:
#         texts = text * i
#         prefill_time, decode_time, hit_rate = model.generate(
#             texts, output_token=args.n_token
#         )
#         prefill_times.append(prefill_time)
#         decode_times.append(decode_time)
#         hit_rates.append(hit_rate)
#         print(
#             f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
#         )
#     utils.plot(
#         batch_sizes,
#         prefill_times,
#         "rmthread-prefill_time-batch_size",
#         "prefill_time(s)",
#         "batch_size",
#     )
#     utils.plot(
#         batch_sizes,
#         decode_times,
#         "rmthread-decode_time-batch_size",
#         "decode_time(s)",
#         "batch_size",
#     )


def test_pp(token_num, batch_size, model):
    n_processed = 0
    while n_processed < token_num:
        n_tokens = min(batch_size, token_num - n_processed)
        tokens = []
        if n_processed == 0:
            tokens.append(1)
        else:
            tokens.append(random.randint(0, model.vocab_size - 1))
        for i in range(1, n_tokens):
            tokens.append(random.randint(0, model.vocab_size - 1))
        input_ids = torch.tensor(tokens, device=model.dev).unsqueeze(0)
        position_ids = (
            torch.arange(
                n_processed,
                n_processed + input_ids.shape[-1],
                dtype=torch.long,
                device=model.dev,
            )
            .unsqueeze(0)
            .view(-1, input_ids.shape[-1])
        )
        one_round_time = time.time()
        logits = model.mixtral_forward(input_ids, position_ids)
        print(f"one_round_time: {time.time()-one_round_time}")
        n_processed += n_tokens
    # if print_flag:
    #     with open(f"../../results/test_pp.txt", "a") as f:
    #         f.write(
    #             f"prefill_time: {exec_time/repeat_num}, token_num:{token_num}, batch_size: {batch_size}, t/s:{token_num*repeat_num/exec_time}\n"
    #         )


def test_tg(token_num, model):
    for j in range(token_num):
        input_id = [random.randint(0, model.vocab_size - 1)]
        input_ids = torch.tensor(input_id, device=model.dev).unsqueeze(0)
        position_ids = (
            torch.arange(
                j,
                j + 1,
                dtype=torch.long,
                device=model.dev,
            )
            .unsqueeze(0)
            .view(-1, input_ids.shape[-1])
        )
        logits = model.mixtral_forward(input_ids, position_ids)
    # if print_flag:
    #     with open(f"../../results/test_tg.txt", "a") as f:
    #         f.write(
    #             f"decode_time: {exec_time/repeat_num}, output_token_num: {token_num}, t/s:{token_num*repeat_num/exec_time}\n"
    # )


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
        "--input",
        type=str,
        default="University of Washington is",
        help="Input text to generate.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=20,
        help="Number of tokens to generate.",
    )
    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width.")
    parser.add_argument(
        "--token_num", type=int, default=128, help="Number of tokens to process."
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat times.")

    args = parser.parse_args()
    data = datasets.load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/raw/main/raw/question.jsonl",
        split="train",
    )
    categories = [
        "writing",
        "roleplay",
        "reasoning",
        "math",
        "coding",
        "extraction",
        "stem",
        "humanities",
    ]
    num_per_category = len(data) // len(categories)
    model = FiddlerMixtral(args)
    model.test_cpu_expert()
    # for i in range(len(categories)):
    #     for j in range(num_per_category):
    #         text = data[i * num_per_category + j]["prompt"][0]
    #         prefill_time, decode_time, hit_rate = model.generate(
    #             [text], output_token=args.n_token
    #         )
    #         print(
    #             f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    #         )
    #     model.write_popular_experts(f"../../results/popularity/{categories[i]}.txt")
    #     model.reset_popular_experts()

    # text = [
    #     "The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle?"
    # ]
    # test_generate(args, text)
    # prefill_time, decode_time, hit_rate = model.generate(
    #     [args.input], output_token=args.n_token
    # )
    # print(
    #     f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    # )

    # if args.token_num > 0:
    #     test_pp(args.token_num, args.batch_size, model)
    # if args.n_token > 0:
    #     test_tg(1, model)

    # for i in range(args.repeat):
    #     torch.cuda.empty_cache()
    #     if args.token_num > 0:
    #         test_pp(args.token_num, args.batch_size, model)
    #     if args.n_token > 0:
    #         test_tg(args.n_token, model)
