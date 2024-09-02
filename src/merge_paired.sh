#! /bin/bash

: '
The script is to be run from the directory containing pairs of fastq
that are intended to be interleaved.
'
# MODIFY: set conda start path
CONDASTARTPATH=/home/lbombini/micromamba/etc/profile.d
# MODIFY: set the conda env name with BBmap installed
CONDAENV="gnn"
# MODIFY: set in sample suffix (e.g. pop_1{SUFFIX}.fastq)
export SUFFIX="_subsampled_20_percent"
# MODIFY: set outdir with a unique name
export OUTDIR="../merged_vk11"

source $CONDASTARTPATH/micromamba.sh
micromamba activate ${CONDAENV}
mkdir -p ${OUTDIR}

interleave_fastq() {
    fwd_path="$1"
    fwd=$(basename "$fwd_path")
    indir=$(dirname "$fwd_path")

    rev="${fwd/_1${SUFFIX}.fastq/_2${SUFFIX}.fastq}"
    rev_path="${indir}/${rev}"

    merged="${fwd/_1${SUFFIX}.fastq/${SUFFIX}_interleaved.fastq}"
    merged_path="${OUTDIR}/${merged}"

    # check if the valid output already exists
    if [ -f "$merged_path" ] && [ -s "$merged_path" ]; then
        echo "${fwd_out} and its pair already exist. Skipping the pair."
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

    echo "Interleaving ${fwd} and ${rev}"
    micromamba run -n gnn reformat.sh in1=$fwd_path in2=$rev_path out=$merged_path
}

export -f interleave_fastq
find . -name "*_1${SUFFIX}.fastq" -type f | parallel interleave_fastq