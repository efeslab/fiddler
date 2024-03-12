import argparse
import os
import utils

from mixtral import FiddlerMixtral


def test_generate(args, text):
    model = FiddlerMixtral(args)
    prefill_times = []
    decode_times = []
    hit_rates = []
    batch_sizes = list(range(1, args.batch_size + 1))
    for i in range(1, args.batch_size + 1):
        texts = text * i
        prefill_time, decode_time, hit_rate = model.generate(
            texts, output_token=args.n_token
        )
        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
        hit_rates.append(hit_rate)
        print(
            f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
        )
    utils.plot(
        batch_sizes,
        prefill_times,
        "beam-prefill_time-batch_size",
        "prefill_time(s)",
        "batch_size",
    )
    utils.plot(
        batch_sizes,
        decode_times,
        "beam-decode_time-batch_size",
        "decode_time(s)",
        "batch_size",
    )


def test_pp(batch_size, prompt_num):
    model = FiddlerMixtral(args)
    text = ["University of Washington is"]
    texts = text * prompt_num
    batch_num = prompt_num // batch_size
    total_prefill_time, total_decode_time = 0, 0
    for i in range(batch_num):
        batched_texts = texts[i * batch_size : (i + 1) * batch_size]
        prefill_time, decode_time, hit_rate = model.generate(
            batched_texts, output_token=args.n_token
        )
        total_prefill_time += prefill_time
        total_decode_time += decode_time

    with open(f"../../results/pp_{batch_size}.txt", "a") as f:
        f.write(f"prefill_time: {total_prefill_time}, decode_time: {total_decode_time}")


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size for inference.",
    )
    parser.add_argument("--beam_num", type=int, default=1, help="Beam search number.")
    parser.add_argument(
        "--prompt_num", type=int, default=1, help="Number of input prompts."
    )

    args = parser.parse_args()

    # model = FiddlerMixtral(args)
    text = ["University of Washington is"]
    # test_generate(args, text)
    # prefill_time, decode_time, hit_rate = model.generate(
    #     [args.input] * args.batch_size, output_token=args.n_token
    # )
    # print(
    #     f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    # )

    test_pp(args.batch_size, args.prompt_num)
