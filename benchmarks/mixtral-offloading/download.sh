#!/bin/bash

mkdir -p Mixtral-8x7B-Instruct-v0.1
cd Mixtral-8x7B-Instruct-v0.1
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/config.json?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/generation_config.json?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00001-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00002-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00003-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00004-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00005-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00006-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00007-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00008-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00009-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00010-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00011-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00012-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00013-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00014-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00015-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00016-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00017-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00018-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model-00019-of-00019.safetensors?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/model.safetensors.index.json?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/special_tokens_map.json?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/tokenizer.json?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/tokenizer.model?download=true
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/tokenizer_config.json?download=true

# rename the files
mv 'config.json?download=true' config.json
mv 'generation_config.json?download=true' generation_config.json
mv 'model-00001-of-00019.safetensors?download=true' model-00001-of-00019.safetensors
mv 'model-00002-of-00019.safetensors?download=true' model-00002-of-00019.safetensors
mv 'model-00003-of-00019.safetensors?download=true' model-00003-of-00019.safetensors
mv 'model-00004-of-00019.safetensors?download=true' model-00004-of-00019.safetensors
mv 'model-00005-of-00019.safetensors?download=true' model-00005-of-00019.safetensors
mv 'model-00006-of-00019.safetensors?download=true' model-00006-of-00019.safetensors
mv 'model-00007-of-00019.safetensors?download=true' model-00007-of-00019.safetensors
mv 'model-00008-of-00019.safetensors?download=true' model-00008-of-00019.safetensors
mv 'model-00009-of-00019.safetensors?download=true' model-00009-of-00019.safetensors
mv 'model-00010-of-00019.safetensors?download=true' model-00010-of-00019.safetensors
mv 'model-00011-of-00019.safetensors?download=true' model-00011-of-00019.safetensors
mv 'model-00012-of-00019.safetensors?download=true' model-00012-of-00019.safetensors
mv 'model-00013-of-00019.safetensors?download=true' model-00013-of-00019.safetensors
mv 'model-00014-of-00019.safetensors?download=true' model-00014-of-00019.safetensors
mv 'model-00015-of-00019.safetensors?download=true' model-00015-of-00019.safetensors
mv 'model-00016-of-00019.safetensors?download=true' model-00016-of-00019.safetensors
mv 'model-00017-of-00019.safetensors?download=true' model-00017-of-00019.safetensors
mv 'model-00018-of-00019.safetensors?download=true' model-00018-of-00019.safetensors
mv 'model-00019-of-00019.safetensors?download=true' model-00019-of-00019.safetensors
mv 'model.safetensors.index.json?download=true' model.safetensors.index.json
mv 'special_tokens_map.json?download=true' special_tokens_map.json
mv 'tokenizer.json?download=true' tokenizer.json
mv 'tokenizer.model?download=true' tokenizer.model
mv 'tokenizer_config.json?download=true' tokenizer_config.json

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
