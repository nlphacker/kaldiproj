#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# Decoding of fMMI or fMPE models (feature-space discriminative training).
# If transform-dir supplied, expects e.g. fMLLR transforms in that dir.

# Begin configuration section.  
iter=final
nj=4
cmd=run.pl
maxactive=7000
beam=13.0
latbeam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
ngselect=2; # Just use the 2 top Gaussians for fMMI/fMPE.  Should match train.
transform_dir=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/decode_fmmi.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/decode_fmmi.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN+LDA+MLLT features."
   echo "You can also use fMLLR features-- you have to supply --transform-dir option."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <transform-dir>                  # where to find fMLLR transforms."
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

model=$srcdir/$iter.mdl

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $srcdir/final.mat $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode_fmmi.sh: no such file $f" && exit 1;
done

feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "Using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne $nj ] && \
     echo "Mismatch in number of jobs with $transform_dir";
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
fi

fmpefeats="$feats fmpe-apply-transform $srcdir/$iter.fmpe ark:- 'ark,s,cs:gunzip -c $dir/gselect.JOB.gz|' ark:- |" 

# Get Gaussian selection info.
$cmd JOB=1:$nj $dir/log/gselect.JOB.log \
  gmm-gselect --n=$ngselect $srcdir/$iter.fmpe "$feats" \
  "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
 gmm-latgen-faster --max-active=$maxactive --beam=$beam --lattice-beam=$latbeam \
   --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $model $graphdir/HCLG.fst "$fmpefeats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --cmd "$cmd" $data $graphdir $dir

exit 0;