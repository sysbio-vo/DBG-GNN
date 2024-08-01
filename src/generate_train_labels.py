import os
import json

data_path = '/home/sysbio/camda2020/camda2020_tiny'

unique_codes = set()

for sample in os.listdir(data_path):
    sample_filename = os.path.basename(sample)
    sample_ext = os.path.splitext(sample_filename)[1]
    if sample_ext not in ('.gz', '.fastq', '.fasta', '.fq', '.fa'):
        continue

    # Example: CAMDA20_MetaSUB_CSD16_BCN_012_1_kneaddata_subsampled_20_percent.fastq
    city_id = list(sample_filename.split('_'))[3]

    unique_codes.add(city_id)

unique_codes_list = list(unique_codes)


labels_mapping = {}
for idx, code in enumerate(unique_codes_list):
    labels_mapping[idx] = code

outfile = '/home/sysbio/camda2020/camda2020_tiny/city_labels.json'

with open(outfile, 'w') as f:
    json.dump(labels_mapping, f)




