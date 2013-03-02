#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#                 Carnegie Mellon University (Author: Yajie Miao)
# Apache 2.0.
# This is a modified version by Yajie for BNF front-end.

# Begin configuration section.  
transform_dir=
iter=
model= # You can specify the model to use (e.g. if you want to use the .alimdl)
nj=4
cmd=run.pl
max_active=7000
beam=15.0
latbeam=7.0
acwt=0.04 # note: only really affects pruning (scoring is on lattices).
min_lmwt=26
max_lmwt=34
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/decode.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/decode.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.alimdl)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
   echo "  --min-lmwt <int>                                 # minumum LM-weight for lattice rescoring "
   echo "  --max-lmwt <int>                                 # maximum LM-weight for lattice rescoring "
   echo "                                                   # speaker-adapted decoding"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$srcdir/final.mdl; 
  else model=$srcdir/$iter.mdl; fi
fi

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode.sh: no such file $f" && exit 1;
done

## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=bnf; fi
echo "$0: feature type is $feat_type"

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

# For the bnf case, this may not be the best manner
case $feat_type in
  bnf) feats="ark,s,cs:splice-feats --left-context=0 --right-context=0 scp:$sdata/JOB/feats.scp ark:- |";;
  lda) feats="ark,s,cs:splice-feats $splice_opts scp:$sdata/JOB/feats.scp ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
##

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
 gmm-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$latbeam \
   --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --cmd "$cmd" --min_lmwt $min_lmwt --max_lmwt $max_lmwt $data $graphdir $dir

exit 0;
