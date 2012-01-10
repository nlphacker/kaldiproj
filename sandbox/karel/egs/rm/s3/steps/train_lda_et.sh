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
# Triphone model training, using cepstral mean normalization plus
# splice-9-frames.  It starts from an existing directory (e.g.
# exp/tri1), supplied as an argument, which is assumed to be built using
# cepstral mean subtraction plus delta features.


if [ $# != 5 ]; then
   echo "Usage: steps/train_lda_mllt.sh <data-dir> <data-subset-dir> <lang-dir> <ali-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_mllt.sh data/train data/lang exp/tri1_ali exp/tri2b"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
datasub=$2
lang=$3
alidir=$4
dir=$5

scripts/is_subset_scp.pl $datasub/feats.scp $data/feats.scp || exit 1;


numiters_et=15 # Before this, update et parameters.
normtype=offset # et option; could be offset [recommended], or none
nutt=15 # Use at most 15 utterances from each speaker for
# estimating transforms, and A and B (we will use all the data
# for estimating the model though, so be careful: we're
# not always using the lists in $dir).

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="5 10 15 20";  
mllt_iters="2 4 6 12";
silphonelist=`cat $lang/silphones.csl`
numiters=25    # Number of iterations of training
maxiterinc=15 # Last iter to increase #Gauss on.
numleaves=1800 # target num-leaves in tree building.
numgauss=$[$numleaves + $numleaves/2];  # starting num-Gauss.
     # Initially mix up to avg. 1.5 Gauss/state ( a bit more
     # than this, due to state clustering... then slowly mix 
     # up to final amount.
totgauss=9000 # Target #Gaussians
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss


mkdir -p $dir

# This variable gets overwritten in this script.
feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/0.mat ark:- ark:- |"
featsub="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$datasub/utt2spk ark:$alidir/cmvn.ark scp:$datasub/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $dir/0.mat ark:- ark:- |"
splicedfeatsub="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$datasub/utt2spk ark:$alidir/cmvn.ark scp:$datasub/feats.scp ark:- | splice-feats ark:- ark:- |"

# compute integer form of transcripts.
scripts/sym2int.pl --ignore-first-field $lang/words.txt < $data/text > $dir/train.tra \
  || exit 1;

echo "Accumulating LDA statistics."

( ali-to-post ark:$alidir/ali ark:- | \
   weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- | \
   acc-lda $alidir/final.mdl "$splicedfeatsub" ark,s,cs:- $dir/lda.acc ) 2>$dir/lda_acc.log
est-lda $dir/0.mat $dir/lda.acc 2>$dir/lda_est.log

cur_lda=$dir/0.mat

echo "Accumulating tree stats"

acc-tree-stats  --ci-phones=$silphonelist $alidir/final.mdl "$feats" \
    ark:$alidir/ali $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


echo "Computing questions for tree clustering"

cat $lang/phones.txt | awk '{print $NF}' | grep -v -w 0 > $dir/phones.list
cluster-phones $dir/treeacc $dir/phones.list $dir/questions.txt 2> $dir/questions.log || exit 1;
scripts/int2sym.pl $lang/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
compile-questions $lang/topo $dir/questions.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

# Have to make silence root not-shared because we will not split it.
scripts/make_roots.pl --separate $lang/phones.txt $silphonelist shared split \
    > $dir/roots.txt 2>$dir/roots.log || exit 1;


echo "Building tree"
build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $lang/topo $dir/tree  2> $dir/train_tree.log || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
   2>$dir/mixup.log || exit 1;

rm $dir/treeacc

# Convert alignments generated from monophone model, to be used as initial alignments.

convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree ark:$alidir/ali ark:$dir/cur.ali 2>$dir/convert.log 
  # Debug step only: convert back and check they're the same.
  convert-ali $dir/1.mdl $alidir/final.mdl $alidir/tree ark:$dir/cur.ali ark:- \
   2>/dev/null | cmp - $alidir/ali || exit 1; 

# Make training graphs
echo "Compiling training graphs"
compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst ark:$dir/train.tra \
    "ark:|gzip -c >$dir/graphs.fsts.gz"  2>$dir/compile_graphs.log  || exit 1;

x=1
while [ $x -lt $numiters ]; do
   echo Pass $x
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
             "ark:gunzip -c $dir/graphs.fsts.gz|" "$feats" \
             ark:$dir/cur.ali 2> $dir/align.$x.log || exit 1;
   fi
   if echo $mllt_iters | grep -w $x >/dev/null; then
     echo "Estimating MLLT"
    ( ali-to-post ark:$dir/cur.ali ark:- | \
       weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
       gmm-acc-mllt --binary=false $dir/$x.mdl "$featsub" ark:- $dir/$x.macc ) 2> $dir/macc.$x.log  || exit 1;

     est-mllt $dir/$x.mat.new $dir/$x.macc 2> $dir/mupdate.$x.log || exit 1;
     gmm-transform-means --binary=false $dir/$x.mat.new $dir/$x.mdl $dir/$[$x+1].mdl 2> $dir/transform_means.$x.log || exit 1;
     compose-transforms --print-args=false $dir/$x.mat.new $cur_lda $dir/$x.mat || exit 1;
     cur_lda=$dir/$x.mat

     feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$data/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $cur_lda ark:- ark:- |"
     featsub="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$alidir/cmvn.ark scp:$datasub/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $cur_lda ark:- ark:- |"
   fi

   gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" ark:$dir/cur.ali $dir/$x.acc 2> $dir/acc.$x.log || exit 1;
   gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   rm $dir/$x.mdl $dir/$x.acc
   rm $dir/$x.occs 
   if [[ $x -le $maxiterinc ]]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done

( cd $dir; rm final.{mdl,occs,mat} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs;
  ln -s `basename $cur_lda` final.mat )

echo Done
