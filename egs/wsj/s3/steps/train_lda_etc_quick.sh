#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation  Arnab Ghoshal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# To be run from ..
# This script trains a model on top of LDA + [something] features, where
# [something] may be MLLT, or ET, or MLLT + SAT.  Any speaker-specific
# transforms are expected to be located in the alignment directory. 
# This script never re-estimates any transforms, it just does model 
# training.  To make this faster, it initializes the model from the
# old system's model, i.e. for each p.d.f., it takes the best-match pdf
# from the old system (based on overlap of tree-stats counts), and 
# uses that GMM to initialize the current GMM.

nj=4
cmd=scripts/run.pl
for x in 1 2; do
  if [ $1 == "--num-jobs" ]; then
     shift
     nj=$1
     shift
  fi
  if [ $1 == "--cmd" ]; then
     shift
     cmd=$1
     shift
  fi  
done

if [ $# != 6 ]; then
   echo "Usage: steps/train_lda_etc_quick.sh <num-leaves> <tot-gauss> <data-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_etc_quick.sh 2500 15000 data/train_si284 data/lang exp/tri3c_ali_si284 exp/tri4c"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

if [ ! -f $alidir/final.mdl -o ! -f $alidir/final.mat ]; then
  echo "Error: alignment dir $alidir does not contain one of final.mdl or final.mat"
  exit 1;
fi

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 15"; # Only realign twice.

oov_sym=`cat $lang/oov.txt`
silphonelist=`cat $lang/silphones.csl`
numiters=20    # Number of iterations of training
maxiterinc=15 # Last iter to increase #Gauss on.
numgauss=$[totgauss/2] # Start with half the total number of Gaussians.  We won't have
  # to mix up much probably, as we're initializing with the old (already mixed-up) pdf's.  
if [ $numgauss -lt $numleaves ]; then numgauss=$numleaves; fi
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss

mkdir -p $dir/log

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data 4
fi

cp $alidir/final.mat $dir/

sifeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk \"ark:cat $alidir/*.cmvn|\" scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"

# featspart[n] gets overwritten later in the script.
for n in `get_splits.pl $nj`; do
  sifeatspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
done

n=`get_splits.pl $nj | awk '{print $1}'`
if [ -f $alidir/$n.trans ]; then
  feats="$sifeats transform-feats --utt2spk=ark:$data/utt2spk \"ark:cat $alidir/*.trans|\" ark:- ark:- |"
  for n in `get_splits.pl $nj`; do
    featspart[$n]="${sifeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"
  done
else
  feats="$sifeats"
  for n in `get_splits.pl $nj`; do featspart[$n]="${sifeatspart[$n]}"; done
fi


# The next stage assumes we won't need the context of silence, which
# assumes something about $lang/roots.txt, but it seems pretty safe.
echo "Accumulating tree stats"
$cmd $dir/log/acc_tree.log \
  acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/*.ali.gz|" $dir/treeacc || exit 1;

echo "Computing questions for tree clustering"
# preparing questions, roots file...
scripts/sym2int.pl $lang/phones.txt $lang/phonesets_cluster.txt > $dir/phonesets.txt || exit 1;
cluster-phones $dir/treeacc $dir/phonesets.txt $dir/questions.txt 2> $dir/log/questions.log || exit 1;
scripts/sym2int.pl $lang/phones.txt $lang/extra_questions.txt >> $dir/questions.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;
scripts/sym2int.pl --ignore-oov $lang/phones.txt $lang/roots.txt > $dir/roots.txt

echo "Building tree"
$cmd $dir/log/train_tree.log \
  build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;

# The gmm-init-model command (with more than the normal # of command-line args)
# will initialize the p.d.f.'s to the p.d.f.'s in the alignment model.
# Note: we first mix-down to $totgauss and then mix-up to $numgauss=$totgauss/2.
# The order of this has nothing to do with the order of the command-line parameters,
# it's how the gmm-mixup program works.  The mix-down phase is to get rid of any
# Gaussians that would be over the final target; the mix-up phase is to get up to
# the initial target.

gmm-init-model  --write-occs=$dir/1.occs  \
  $dir/tree $dir/treeacc $lang/topo $dir/tmp.mdl $alidir/tree $alidir/final.mdl  \
  2>$dir/log/init_model.log || exit 1;

gmm-mixup --mix-down=$totgauss --mix-up=$numgauss $dir/tmp.mdl $dir/1.occs $dir/1.mdl \
   2>> $dir/log/init_model.log || exit 1;

rm $dir/tmp.mdl $dir/treeacc

# Convert alignments in $alidir, to use as initial alignments.
# This assumes that $alidir was split in the same number of pieces
# as the current dir.

echo "Converting old alignments"
for n in `get_splits.pl $nj`; do # do this locally; it's fast.
  convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
   "ark:gunzip -c $alidir/$n.ali.gz|" "ark:|gzip -c >$dir/$n.ali.gz" \
    2>$dir/log/convert$n.log  || exit 1;
done

# Make training graphs (this is split in 4 parts).
echo "Compiling training graphs"
rm $dir/.error 2>/dev/null
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/compile_graphs$n.log \
    compile-train-graphs --batch-size=750 $dir/tree $dir/1.mdl  $lang/L.fst  \
      "ark:sym2int.pl --map-oov \"$oov_sym\" --ignore-first-field $lang/words.txt < $data/split$nj/$n/text |" \
      "ark:|gzip -c >$dir/$n.fsts.gz" || touch $dir/.error &
done
wait;
[ -f $dir/.error ] && echo "Error compiling training graphs" && exit 1;

x=1
while [ $x -lt $numiters ]; do
   echo Pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     for n in `get_splits.pl $nj`; do
       $cmd $dir/log/align.$x.$n.log \
         gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/$x.mdl \
           "ark:gunzip -c $dir/$n.fsts.gz|" "${featspart[$n]}" \
           "ark:|gzip -c >$dir/$n.ali.gz" || touch $dir/.error &
     done
     wait;
     [ -f $dir/.error ] && echo "Error aligning data on iteration $x" && exit 1;
   fi
   for n in `get_splits.pl $nj`; do
     $cmd $dir/log/acc.$x.$n.log \
       gmm-acc-stats-ali --binary=false $dir/$x.mdl "${featspart[$n]}" \
         "ark:gunzip -c $dir/$n.ali.gz|" $dir/$x.$n.acc || touch $dir/.error &
   done
   wait;
   [ -f $dir/.error ] && echo "Error accumulating stats on iteration $x" && exit 1;
   $cmd $dir/log/update.$x.log \
     gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
       "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
   rm $dir/$x.mdl $dir/$x.*.acc
   rm $dir/$x.occs 
   if [[ $x -le $maxiterinc ]]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done

if [ "$feats" != "$sifeats" ]; then
  # we have speaker-specific transforms, so need to estimate an alignment model.
  # Accumulate stats for "alignment model" which is as the model but with
  # the default features (shares Gaussian-level alignments).
  echo "Estimating alignment model."
  for n in `get_splits.pl $nj`; do
    $cmd $dir/acc_alimdl.$n.log \
      ali-to-post "ark:gunzip -c $dir/$n.ali.gz|" ark:-  \| \
        gmm-acc-stats-twofeats $dir/$x.mdl "${featspart[$n]}" "${sifeatspart[$n]}" \
          ark:- $dir/$x.$n.acc2 || touch $dir/.error &
  done
  wait;
  [ -f $dir/.error ] && echo "Error accumulating alignment statistics." && exit 1;
  # Update model.
  $cmd $dir/log/est_alimdl.log \
    gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc2|" $dir/$x.alimdl || exit 1;
  rm $dir/$x.*.acc2
  ( cd $dir; rm final.alimdl 2>/dev/null; ln -s $x.alimdl final.alimdl )
fi


( cd $dir; rm final.mdl 2>/dev/null; ln -s $x.mdl final.mdl )

echo Done
