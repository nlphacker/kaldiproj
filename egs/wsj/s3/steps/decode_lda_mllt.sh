#!/bin/bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

# Apache 2.0

# Decoding script that works with a GMM model and the baseline
# [e.g. MFCC] features plus cepstral mean subtraction plus
# LDA+MLLT or similar transform.
# This script just generates lattices for a single broken-up
# piece of the data.

if [ -f ./path.sh ]; then . ./path.sh; fi

numjobs=1
jobid=0

for x in `seq 3`; do
  if [ "$1" == "-j" ]; then
    shift;
    numjobs=$1;
    jobid=$2;
    shift; shift;
  fi
done


if [ $# != 3 ]; then
   echo "Usage: steps/decode_lda_mllt.sh [-j num-jobs job-number] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_lda_mllt.sh -j 8 0 exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_dev93_tgpr"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $numjobs -gt 1 ]; then
  mydata=$data/split$numjobs/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $srcdir/final.mat $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "decode_lda_mllt.sh: no such file $f";
     exit 1;
  fi
done


# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 \
  --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.$jobid.gz" \
     2> $dir/decode$jobid.log || exit 1;

