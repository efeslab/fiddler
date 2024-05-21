import pandas as pd

# csv_path = "/home/yilegu/fiddler/benchmarks/results/eval-2024-05-18-09-30-14-MII.csv"
# prefill_csv_path = "/home/yilegu/fiddler/benchmarks/results/eval-2024-05-21-20-43-01-MII-prefill.csv"
csv_path = "/home/yilegu/fiddler/benchmarks/results/eval-2024-05-19-18-12-19-Mixtral-Offload.csv"
prefill_csv_path = "/home/yilegu/fiddler/benchmarks/results/eval-2024-05-21-20-14-25-Mixtral-Offload-prefill.csv"
output_csv_path = "/home/yilegu/fiddler/benchmarks/results/Mixtral-Offload-merge.csv"

df = pd.read_csv(csv_path)
df_prefill = pd.read_csv(prefill_csv_path)

f = open(output_csv_path, 'w')
f.write('input_token, output_token, batch_size, prefill_time, decode_time\n')

input_lengths = [32, 64, 128, 256, 512]
output_lengths = [64, 128, 256, 512]
batch_sizes = [1]

for input_length in input_lengths:
    for output_length in output_lengths:
        for batch_size in batch_sizes:
            df_filtered = df[(df['input_token'] == input_length) & (df['output_token'] == output_length) & (df['batch_size'] == batch_size)]
            df_prefill_filtered = df_prefill[(df_prefill['input_token'] == input_length) & (df_prefill['output_token'] == 1) & (df_prefill['batch_size'] == batch_size)]
            assert len(df_filtered) == 1 and len(df_prefill_filtered) == 1
            total_time = df_filtered['time'].values[0]
            prefill_time = df_prefill_filtered['time'].values[0]
            f.write(f'{input_length}, {output_length}, {batch_size}, {prefill_time}, {total_time - prefill_time}\n')
            
            