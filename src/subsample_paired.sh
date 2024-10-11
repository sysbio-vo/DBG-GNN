#! /bin/bash

# MODIFY: set your paths here
CONDAPATH=/home/nepotlet/miniconda3/bin
CONDASTARTPATH=/home/nepotlet/miniconda3/etc/profile.d
source $CONDASTARTPATH/conda.sh


# MODIFY: activate env with seqkit
echo "activating conda env"
conda activate kneaddata

# INPUTS
export INDIR=/scratch/sysbio/camda2020/camda2020/fastp_vk11 # TODO: use complete data
export PROPORTION_TO_TAKE=0.2 # MODIFY: set proportion of reads to subsample
export OUTDIR=/scratch2/sysbio/qced_data # MODIFY: set outdir with a unique name
export PAIRED_SAMPLENAME_SUFFIX=_trimmed


export PROPORTION_INT=$(echo "$PROPORTION_TO_TAKE * 100 / 1" | bc)
mkdir -p $OUTDIR

# MODIFY: (optionally) 
# set datetime after which downloaded samples will be ignored 
# (to not process partially downloaded files)
# export DATETIME="2024-08-08 00:00:00"

process_sample() {
    fwd_path="$1"
    fwd=$(basename "$fwd_path")
    fwd_out=${OUTDIR}/${fwd%.fastq.gz}_subsampled_${PROPORTION_INT}_percent.fastq

    dir=$(dirname "$fwd_path")

    rev="${fwd/_1${PAIRED_SAMPLENAME_SUFFIX}.fastq.gz/_2${PAIRED_SAMPLENAME_SUFFIX}.fastq.gz}"
    rev_path="${dir}/${rev}"
    rev_out=${OUTDIR}/${rev%.fastq.gz}_subsampled_${PROPORTION_INT}_percent.fastq

    # check if the valid output already exists
    if [ -f "$fwd_out" ] && [ -s "$fwd_out" ] && [ -f "$rev_out" ] && [ -s "$rev_out" ]; then
        echo "${fwd_out} and its pair already exist. Skipping the pair."
        return 0
    fi

    # check if the pair exists
    if [ ! -f "$rev_path" ]; then
        echo "R2 for ${fwd_path} has not been found. Skipping the pair"
        return 1
    fi

    # check if not empty
    if [ ! -s "$fwd_path" ] || [ ! -s "$rev_path" ]; then
        echo "Either $fwd_path or its pair is empty. Skipping the pair"
        return 1
    fi

    seed=$RANDOM
    echo "Subsampling $fwd_path  Proportion: $PROPORTION_TO_TAKE  Seed: ${seed}"
    seqkit sample -p $PROPORTION_TO_TAKE $fwd_path -o $fwd_out -s $seed
    
    echo "Subsampling $rev_path  Proportion: $PROPORTION_TO_TAKE  Seed: ${seed}"
    seqkit sample -p $PROPORTION_TO_TAKE $rev_path -o $rev_out -s $seed
}

export -f process_sample

# process ${PAIRED_SAMPLENAME_SUFFIX}.fastq.gz files from the input dir
echo "processing samples from $INDIR"
find $INDIR -name "*_1${PAIRED_SAMPLENAME_SUFFIX}.fastq.gz" -type f | parallel process_sample
