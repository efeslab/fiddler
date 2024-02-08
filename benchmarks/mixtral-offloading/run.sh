#!/bin/bash

conda create -n mixtral-offloading python=3.10
conda activate mixtral-offloading
pip install -r requirements.txt
./download.sh
python3 eval.py