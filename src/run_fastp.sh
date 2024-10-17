#! /bin/bash

# MODIFY: set your paths here
CONDAPATH=/home/nepotlet/miniconda3/bin
CONDASTARTPATH=/home/nepotlet/miniconda3/etc/profile.d
source $CONDASTARTPATH/conda.sh

echo "activating conda env"
CONDAENV="gnn"
conda activate $CONDAENV

# INPUTS
export INDIR="/scratch/sysbio/camda2020/camda2020"
export OUTDIR="/scratch2/sysbio/fastped_data"
export OUTFILE_SUFFIX="_trimmed"
export SAMPLE_FILENAME_PATTERN='*_1.fastq.gz'

mkdir -p $OUTDIR

run_fastp() {
    fwd_path="$1"
    fwd=$(basename "$fwd_path")
    indir=$(dirname "$fwd_path")

    rev="${fwd/_1.fastq.gz/_2.fastq.gz}"
    rev_path="${indir}/${rev}"

    fwd_trimmed="${fwd/_1.fastq.gz/_1${OUTFILE_SUFFIX}.fastq.gz}"
    fwd_trimmed_path="${OUTDIR}/${fwd_trimmed}"
    rev_trimmed="${rev/_2.fastq.gz/_2${OUTFILE_SUFFIX}.fastq.gz}"
    rev_trimmed_path="${OUTDIR}/${rev_trimmed}"

    # check if the valid output already exists
    if [ -f "$fwd_trimmed_path" ] && [ -s "$fwd_trimmed_path" ] && [ -f "$rev_trimmed_path" ] && [ -s "$rev_trimmed_path" ]; then
        echo "${fwd_trimmed_path} and its pair already exist. Skipping the pair."
        return 0
    fi

    # check if the pair exists
    if [ ! -f "$rev" ]; then
        echo "R2 for ${fwd} has not been found. Skipping the pair"
        return 1
    fi

    # check if not empty
    if [ ! -s "$fwd" ] || [ ! -s "$rev" ]; then
        echo "Either $fwd or its pair is empty. Skipping the pair"
        return 1
    fi

    echo "Running fastp on ${fwd_path} and ${rev_path}"
    conda run -n $CONDAENV fastp -i $fwd_path -I $rev_path -o $fwd_trimmed_path -O $rev_trimmed_path

}

export -f run_fastp
echo "processing samples with fastp from $INDIR"
find $INDIR -name "$SAMPLE_FILENAME_PATTERN" -type f | parallel run_fastp