#! /bin/bash

PROPORTION_TO_TAKE=0.2

mkdir -p filtered_subsampled/

for filtered_sample in filtered_human_reads/*_kneaddata.fastq; do
	filename=${filtered_sample#*/}
	basename=${filename%.fastq}
	
	seqkit sample -p $PROPORTION_TO_TAKE $filtered_sample -o filtered_subsampled/${basename}_subsampled_20_percent.fastq
done
