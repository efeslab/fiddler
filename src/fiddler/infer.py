import argparse
import os

# import utils
import random
from mixtral import FiddlerMixtral
import torch
import time
import datasets


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
    model = FiddlerMixtral(args)
    prefill_time, decode_time, hit_rate = model.generate(
        [args.input], output_token=args.n_token
    )
    print(
        f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    )
