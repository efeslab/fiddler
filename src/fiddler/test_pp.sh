#!/bin/bash

# Define an array of batch sizes
batch_sizes=(8 16 32 64)

# Loop through each batch size
for batch_size in "${batch_sizes[@]}"; do
    echo "batch_size: $batch_size"
    python3 infer.py --batch-size "$batch_size" --beam_num=2 --n-token=32 --prompt_num=64
done
