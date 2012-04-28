#!/bin/bash

# Copyright Daniel Povey,  2012.  
# Apache 2.0.

# Train a diagonal mixture of Gaussians.  This is trained without
# reference to class labels-- except that, optionally, you can down-weight
# silence phones, and alignments are needed for that.
#
# The current use for this is in fMMI training.

# Begin configuration section.
nj=4
cmd=run.pl
num_iters=3
silence_weight=
stage=-2
# The value "intermediate" is a number of Gaussians we first obtain by clustering
# the Gaussians within each state of the model, before clustering down to
# $num_Gauss.  This is for efficiency.  It's not a very important parameter,
# as far as I know.
intermediate=2000
num_gselect=50 # Number of Gaussian-selection indices to use while training
               # the model.
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: steps/train_ubm_lda_etc.sh <num-gauss> <data> <lang> <alignment-dir|src-dir> <dir>"
  echo " e.g.: steps/train_ubm_lda_etc.sh 400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c"
  echo "Options: "
  echo "  --silence-weight <sil-weight>                  # default 1.0.  Use to down-weight silence."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <num-job>                                 # number of parallel jobs to run."
  echo "  --num-iters <niter>                            # number of iterations of training (default: $num_iters)"
  echo "  --stage <stage>                                # stage to do partial re-run from."
  exit 1;
fi

num_gauss=$1
data=$2
lang=$3
alidir=$4
dir=$5

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

sdata=$data/split$nj
mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

[ -f $alidir/1.trans ] && echo Using transforms from $alidir && \
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk $alidir/JOB.trans ark:- ark:- |"

if [ ! -z "$silence_weight" ]; then
  [ ! -f $alidir/ali.1.gz ] && \
    echo "You specified weighting for silence but $alidir/ali.1.gz does not exist." && exit 1;
  [ "$nj" -ne "`cat $alidir/num_jobs`" ] && \
    echo "You specified silence weight but $alidir has different #jobs." && exit 1;
  weights="--weights='ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- | weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- |'"
else
  weights=
fi

# $intermediate should be more than $num_gauss..
[ $[$num_gauss*2] -gt $intermediate ] && intermediate=$[$num_gauss*2] \
  && echo "Setting intermediate=$intermediate (it was too small)";

echo "Clustering Gaussians in $alidir/final.mdl"

if [ $stage -ge -2 ]; then
 $cmd $dir/log/cluster.log \
  init-ubm --fullcov-ubm=false --intermediate-numcomps=$intermediate \
    --ubm-numcomps=$num_gauss $alidir/final.mdl $alidir/final.occs $dir/0.dubm   || exit 1;
fi

# Store Gaussian selection indices on disk-- this speeds up the training passes.
if [ $stage -ge -1 ]; then
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $dir/0.dubm "$feats" \
      "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
done
wait
[ -f $dir/.error ] && echo "Error doing GMM selection" && exit 1;

for x in `seq 0 $[$num_iters-1]`; do
  echo "Training pass $x"
  # Accumulate stats.
  $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
    gmm-global-acc-stats $weights "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" \
      $dir/$x.dubm "$feats" $dir/$x.JOB.acc || exit 1;
  if [ $x -lt $[$num_iters-1] ]; then # Don't remove low-count Gaussians till last iter,
    opt="--remove-low-count-gaussians=false" # or gselect info won't be valid any more.
  fi
  $cmd $dir/log/update.$x.log \
    gmm-global-est $opt $dir/$x.dubm "gmm-global-sum-accs - $dir/$x.*.acc|" \
    $dir/$[$x+1].dubm || exit 1;
  rm $dir/$x.*.acc $dir/$x.dubm
done

rm $dir/gselect.*.gz
mv $dir/$num_iters.dubm $dir/final.dubm || exit 1;
exit 0;
