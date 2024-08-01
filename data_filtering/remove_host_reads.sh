#! /bin/bash

mkdir -p filtered_human_reads/

for sample in *.fastq.gz; do
	kneaddata --unpaired $sample --reference-db ~/GRCh38/kneaddata/ --output filtered_human_reads --trimmomatic ~/trimmomatic/Trimmomatic-0.39/ --threads 16
done
