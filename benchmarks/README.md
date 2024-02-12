# Benchmarks

Download the ShareGPT dataset before running benchmarks.

```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

- `latency.py`: Single batch latency evaluation
- `microbench.py`: Microbenchmarks

## Baseline evaluations
Following is the instructions for running evaluation for two baselines: [Mixtral offloading](https://github.com/dvmazur/mixtral-offloading) and [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)

### DeepSpeed-MII
Follow these steps to install DeepSpeed packages and run evaluation.

```
conda create -n deepspeed python=3.10
conda activate deepspeed
conda install -c anaconda mpi4py
pip install deepspeed-mii accelerate
python3 eval-baseline.py --framework=deepspeed-mii
```

### Mixtral Offloading

Follow these steps to install packages, download model and datasets, and run evaluation.

```
conda create -n mixtral-offload python=3.10
conda activate mixtral-offload
pip install -r requirements.txt
./download.sh
python3 eval-baseline.py --framework=mixtral-offloading
```
