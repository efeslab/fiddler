# Mixtral offloading

This is a modified version of Mixtral offloading to enable running original Mixtral8x7B fp16 version in local setting.

## Performance Evaluate

Follow the these steps to build the model and evaluate performance.

```
pip install -r requirements.txt
./download.sh
python3 eval.py > output.log
```
