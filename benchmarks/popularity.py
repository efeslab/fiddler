import argparse
import json
import os
import sys


from datasets import load_dataset
from tqdm import tqdm

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
        "--n-token",
        type=int,
        default=1,
        help="Number of tokens to generate.",
    )
    parser.add_argument("--beam-width", type=int, default=1, help="Beam search width.")
    parser.add_argument("--profile-popularity", action="store_true", default=True, help="Profile expert popularity.")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=['mmlu'], help="Dataset name.")

    args = parser.parse_args()
    model = FiddlerMixtral(args)
    if args.dataset == "mmlu":
        for subject in ['high_school_us_history', 'high_school_biology', 'college_computer_science', 'machine_learning', 'college_medicine', 'international_law', 'high_school_mathematics']:
            dataset = load_dataset("lukaemon/mmlu", subject, split="test")
            print(f'Test subject {subject} with {len(dataset)} samples.')
            total_input_tokens = 0
            for example in tqdm(dataset, desc="Processing dataset"):
                prefill_time, decode_time, hit_rate = model.generate(
                    example['input'], output_token=args.n_token
                )
                total_input_tokens += len(example['input'].split())
            expert_popularity = model.get_expert_popularity()
            result = {}
            result["total_input_tokens"] = total_input_tokens
            result["expert_popularity"] = expert_popularity
            # save to json
            with open(f"expert_popularity_{subject}.json", "w") as f:
                json.dump(result, f)
            model.reset_expert_popularity()