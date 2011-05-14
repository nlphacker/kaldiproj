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


# tri2l is splice-9-frames + LDA + MLLT/STC, + SAT
#
# See tri3f for a full-size (si-284) system that builds from this.

if [ -f path.sh ]; then . path.sh; fi

dir=exp/tri2l
srcdir=exp/tri1
srcmodel=$srcdir/final.mdl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

numiters=35
maxiterinc=20 # By this iter, we have all the Gaussians.
realign_iters="10 20 30"; 
mllt_iters="2 4 6 8"; 
fmllr_iters="9 14 19";
randprune=5.0 # Use about 1/5 of frames.. 
numleaves=2000
numgauss=2000 # initial num-gauss smallish so that transform-training
              # code (when we modify this script) is a bit faster.
totgauss=10000 # Total num-gauss
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss

silphonelist=`cat data/silphones.csl`

mkdir -p $dir
cp $srcdir/train.scp $dir
cp $srcdir/train.tra $dir

scripts/filter_scp.pl $dir/train.scp data/train.utt2spk > $dir/train.utt2spk
scripts/utt2spk_to_spk2utt.pl $dir/train.utt2spk > $dir/train.spk2utt

scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.scp
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.tra
scripts/split_scp.pl --utt2spk=$dir/train.utt2spk $dir/train{,1,2,3}.utt2spk

for n in 1 2 3 ""; do # The "" handles the un-split one.  Creating spk2utt files..
  scripts/utt2spk_to_spk2utt.pl $dir/train$n.utt2spk > $dir/train$n.spk2utt
done

srcfeats="ark,s,cs:add-deltas --print-args=false scp:$dir/train.scp ark:- |"
feats="ark,s,cs:splice-feats --print-args=false scp:$dir/train.scp ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
for n in 1 2 3; do
   srcfeatspart[$n]="ark,s,cs:add-deltas --print-args=false scp:$dir/train${n}.scp ark:- |"
   rawfeatspart[$n]="ark,s,cs:splice-feats --print-args=false scp:$dir/train${n}.scp ark:- |"
   featspart[$n]="ark,s,cs:splice-feats --print-args=false scp:$dir/train${n}.scp ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
done

cp $srcdir/topo $dir

# Align all training data using old model (and old graphs, since we
# use the same data-subset as last time). 

echo "Aligning all training data"

rm -f $dir/.error
for n in 1 2 3; do
   gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $srcmodel \
       "ark:gunzip -c $srcdir/graphs${n}.fsts.gz|" "${srcfeatspart[$n]}" \
       "ark:|gzip -c >$dir/0.${n}.ali.gz" \
           2> $dir/align.0.${n}.log || touch $dir/.error &
done
wait;
[ -f $dir/.error ] &&  echo alignment error RE old system && exit 1

# Now accumulate LDA stats.  

for n in 1 2 3; do
 ( ali-to-post "ark:gunzip -c $dir/0.$n.ali.gz|" ark:- | \
    weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
    acc-lda $srcmodel "${rawfeatspart[$n]}" \
     ark:- $dir/lda$n.acc ) 2>$dir/lda_acc.$n.log || touch $dir/.error &
done
wait

[ -f $dir/.error ] &&  echo lda-acc error && exit 1

est-lda $dir/lda.mat $dir/lda?.acc  2>$dir/lda_est.log || exit 1
rm $dir/lda?.acc

acc-tree-stats  --ci-phones=$silphonelist $srcmodel "$feats" "ark:gunzip -c $dir/0.?.ali.gz|" $dir/treeacc 2> $dir/acc.tree.log  || exit 1;


# The next few commands are involved with making the questions
# for tree clustering.  The extra complexity vs. the RM recipe has
# to do with the desire to ask questions about
# ignoring things like stress and position-in-word, and ask questions
# separately about stress and position-in-word.

# Don't include silences as things to be clustered -> --nosil option.
scripts/make_shared_phones.sh --nosil | scripts/sym2int.pl data/phones.txt > $dir/phone_sets.list
cluster-phones $dir/treeacc $dir/phone_sets.list $dir/questions.txt 2> $dir/cluster_phones.log || exit 1;
scripts/int2sym.pl data/phones.txt < $dir/questions.txt > $dir/questions_syms.txt
scripts/make_extra_questions.sh | cat $dir/questions_syms.txt - > $dir/questions_syms_all.txt
scripts/sym2int.pl data/phones.txt < $dir/questions_syms_all.txt > $dir/questions_all.txt

compile-questions $dir/topo $dir/questions_all.txt $dir/questions.qst 2>$dir/compile_questions.log || exit 1;

scripts/make_roots.sh > $dir/roots_syms.txt
scripts/sym2int.pl --ignore-oov data/phones.txt  < $dir/roots_syms.txt > $dir/roots.txt


build-tree --verbose=1 --max-leaves=$numleaves \
    $dir/treeacc $dir/roots.txt \
    $dir/questions.qst $dir/topo $dir/tree  2> $dir/train_tree.log || exit 1;

gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $dir/topo $dir/1.mdl 2> $dir/init_model.log || exit 1;

gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl \
    2>$dir/mixup.log || exit 1;


rm $dir/treeacc $dir/1.occs


# Convert alignments generated from previous model, to use as initial alignments.

for n in 1 2 3; do
  convert-ali  $srcmodel $dir/1.mdl $dir/tree \
      "ark:gunzip -c $dir/0.$n.ali.gz|" \
      "ark:|gzip -c > $dir/cur$n.ali.gz" 2>$dir/convert.$n.log 
done
rm $dir/0.?.ali.gz

# Make training graphs
echo "Compiling training graphs"

rm -f $dir/.error
for n in 1 2 3; do
  compile-train-graphs $dir/tree $dir/1.mdl  data/L.fst ark:$dir/train${n}.tra \
     "ark:|gzip -c > $dir/graphs${n}.fsts.gz" \
     2>$dir/compile_graphs.${n}.log || touch $dir/.error &
done

wait
[ -f $dir/.error ] &&  echo compile-graphs error && exit 1

cur_lda=$dir/lda.mat
x=1
while [ $x -lt $numiters ]; do
   echo "Pass $x"
   if echo $realign_iters | grep -w $x >/dev/null; then
     echo "Aligning data"
     rm -f $dir/.error
     for n in 1 2 3; do
       gmm-align-compiled $scale_opts --beam=8 --retry-beam=40 $dir/$x.mdl \
           "ark:gunzip -c $dir/graphs${n}.fsts.gz|" "${featspart[$n]}" \
           "ark:|gzip -c >$dir/cur${n}.ali.gz" 2> $dir/align.$x.$n.log \
             || touch $dir/.error &
     done
     wait 
     [ -f $dir/.error ] && echo error aligning data && exit 1
   fi
   if echo $fmllr_iters | grep -w $x >/dev/null; then
     for n in 1 2 3; do
       sifeatsn="ark:splice-feats scp:$dir/train$n.scp ark:- | transform-feats $cur_lda ark:- ark:-|"
       ( ali-to-post "ark:gunzip -c $dir/cur$n.ali.gz|" ark:- | \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
        gmm-post-to-gpost $dir/$x.mdl "${featspart[$n]}" ark:- ark:- | \
        gmm-est-fmllr-gpost --spk2utt=ark:$dir/train$n.spk2utt $dir/$x.mdl "$sifeatsn" ark,s,cs:- ark:$dir/tmp$n.trans ) \
           2> $dir/trans.$x.$n.log || touch .error &
     done
     wait
     [ -f $dir/.error ] && echo error computing fMLLR transforms && exit 1
     
     feats="ark:splice-feats scp:$dir/train.scp ark:- | transform-feats $cur_lda ark:- ark:- | transform-feats --utt2spk=ark:$dir/train.utt2spk \"ark:cat $dir/cur?.trans|\" ark:- ark:- |"
     for n in 1 2 3; do
       mv $dir/tmp$n.trans $dir/cur$n.trans || exit 1
       featspart[$n]="ark:splice-feats scp:$dir/train.scp ark:- | transform-feats $cur_lda ark:- ark:- | transform-feats --utt2spk=ark:$dir/train$n.utt2spk ark:$dir/cur$n.trans ark:- ark:- |"
     done
   fi
   if echo $mllt_iters | grep -w $x >/dev/null; then # Do MLLT update.
     # Use the 1st of the three subsets... no need to use more as few params to update
     for n in 1 2 3; do 
      ( ali-to-post "ark:gunzip -c $dir/cur$n.ali.gz|" ark:- | \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- | \
        gmm-acc-mllt --rand-prune=$randprune --binary=false $dir/$x.mdl \
          "${featspart[$n]}" ark:- $dir/$x.$n.macc ) 2> $dir/macc.$x.$n.log \
          || touch $dir/.error &
     done
     wait
     [ -f $dir/.error ] && echo error accumulating MLLT stats && exit 1

     est-mllt $dir/$x.mat.new $dir/$x.?.macc 2> $dir/mupdate.$x.log || exit 1;
     rm $dir/$x.?.macc
     gmm-transform-means --binary=false $dir/$x.mat.new $dir/$x.mdl $dir/$[$x+1].mdl 2> $dir/transform_means.$x.log || exit 1;
     compose-transforms --print-args=false $dir/$x.mat.new $cur_lda $dir/$x.mat || exit 1;

     cur_lda=$dir/$x.mat
     feats="ark,s,cs:splice-feats --print-args=false scp:$dir/train.scp ark:- | transform-feats $cur_lda ark:- ark:- |" 
     for n in 1 2 3; do
       featspart[$n]="ark,s,cs:splice-feats --print-args=false scp:$dir/train$n.scp ark:- | transform-feats $cur_lda ark:- ark:- |" 
     done
   else # do GMM update
     gmm-acc-stats-ali --binary=false $dir/$x.mdl "$feats" \
     "ark,s,cs:gunzip -c $dir/cur?.ali.gz|" $dir/$x.acc 2> $dir/acc.$x.log  || exit 1;
     gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl $dir/$x.acc $dir/$[$x+1].mdl 2> $dir/update.$x.log || exit 1;
   fi
   rm $dir/$x.mdl $dir/$x.acc $dir/$x.occs 2>/dev/null
   
   if [ $x -le $maxiterinc ]; then 
      numgauss=$[$numgauss+$incgauss];
   fi
   x=$[$x+1];
done


( cd $dir; rm final.{mdl,mat} 2>/dev/null; 
  ln -s $x.mdl final.mdl; 
  ln -s `basename $cur_lda` final.mat )

defaultfeats="ark:splice-feats scp:$dir/train.scp ark:- | transform-feats $cur_lda ark:- ark:-|"

# Accumulate stats for "alignment model" which is as the model but with
# the unadapted, default features (shares Gaussian-level alignments).
( ali-to-post "ark:cat $dir/cur?.ali.gz|" ark:-  | \
  gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$defaultfeats" ark:- $dir/$x.acc2 ) 2>$dir/acc_alimdl.log || exit 1;
  # Update model.
  gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false \
    $dir/$x.mdl $dir/$x.acc2 $dir/$x.alimdl \
    2>$dir/est_alimdl.log  || exit 1;
 rm $dir/$x.acc2
 
( cd $dir; rm final.{mdl,alimdl,mat} 2>/dev/null; 
  ln -s $x.mdl final.mdl; ln -s $x.alimdl final.alimdl
  ln -s `basename $cur_lda` final.mat )
