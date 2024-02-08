# Benchmarks
This page contains instructions for running evaluation for two baselines: [Mixtral offloading](https://github.com/dvmazur/mixtral-offloading) and [DeepSpeed-Mii](https://github.com/microsoft/DeepSpeed-MII)

## DeepSpeed-Mii
Follow these steps to install DeepSpeed packages and run evaluation.

```
conda create -n deepspeed python=3.10
conda activate deepspeed
conda install -c anaconda mpi4py
pip install deepspeed-mii accelerate
python3 eval.py --framework=deepspeed-mii
```

## Mixtral Offloading

Follow these steps to install packages, download model and datasets, and run evaluation.

```
conda create -n mixtral-offload python=3.10
conda activate mixtral-offload
pip install -r requirements.txt
./download.sh
python3 eval.py --framework=mixtral-offloading
```
