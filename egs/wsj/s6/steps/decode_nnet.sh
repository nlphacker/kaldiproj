#!/bin/bash

# Copyright 2012  Karel Vesely, Daniel Povey
# Apache 2.0

# Begin configuration section.  
nnet= # You can specify the nnet to use (e.g. if you want to use the .alinnet)
feature_transform= # You can specify the feature transform to use for the feedforward
model= # You can specify the transition model to use (e.g. if you want to use the .alimdl)
class_frame_counts= # You can specify frame count to compute PDF priors 

nj=4
cmd=run.pl
max_active=7000
beam=13.0 # GMM:13.0
latbeam=8.0 # GMM:6.0
acwt=0.10 # GMM:0.0833, note: only really affects pruning (scoring is on lattices).
scoring_opts="--min-lmwt 4 --max-lmwt 15"
score_args=
use_gpu_id=-1 #disable gpu
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: $0 exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --nnet <nnet>                                    # which nnet to use (e.g. to"
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.nnet)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "                                                   # speaker-adapted decoding"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z "$nnet" ]; then # if --nnet <nnet> was not specified on the command line...
  nnet=$srcdir/final.nnet; 
fi
[ -z "$nnet" ] && echo "Error nnet '$nnet' does not exist!" && exit 1;

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  model=$srcdir/final.mdl;
fi

# find the feature_transform to use
if [ -z "$feature_transform" ]; then
  feature_transform=$srcdir/final.feature_transform
fi
if [ ! -f $feature_transform ]; then
  echo "Missing feature_transform '$feature_transform'"
  exit 1
fi

# check that files exist
for f in $sdata/1/feats.scp $nnet_i $nnet $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# PREPARE THE LOG-POSTERIOR COMPUTATION PIPELINE
if [ -z "$class_frame_counts" ]; then
  class_frame_counts=$srcdir/ali_train_pdf.counts
else
  echo "Overriding class_frame_counts by $class_frame_counts"
fi

# Create the feature stream:
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# Optionally add cmvn
if [ -f $srcdir/norm_vars ]; then
  norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
  [ ! -f $sdata/1/cmvn.scp ] && echo "$0: cannot find cmvn stats $sdata/1/cmvn.scp" && exit 1
  feats="$feats apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
fi
# Optionally add deltas
if [ -f $srcdir/delta_order ]; then
  delta_order=$(cat $srcdir/delta_order)
  feats="$feats add-deltas --delta-order=$delta_order ark:- ark:- |"
fi


# Run the decoding in the queue
$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  nnet-forward --feature-transform=$feature_transform --no-softmax=true --class-frame-counts=$class_frame_counts --use-gpu-id=$use_gpu_id $nnet "$feats" ark:- \| \
  latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam \
  --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $model $graphdir/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

# Run the scoring
[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh $score_args $scoring_opts --cmd "$cmd" $data $graphdir $dir 2>$dir/scoring.log || exit 1;

exit 0;
