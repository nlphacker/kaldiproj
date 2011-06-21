#!/bin/bash

# Copyright 2010-2011 	Microsoft Corporation,  Saarland University

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

# deocde_tri_regtree_mllr.sh is as ../decode_tri.sh but estimating MLLR in test,
# per speaker.  There is no SAT.  Use a regression-tree with top-level speech/sil
# split (no silence weighting).

if [ -f path.sh ]; then . path.sh; fi
srcdir=exp/decode_tri1
dir=exp/decode_tri1_fmllr+regtree_mllr
mkdir -p $dir
model=exp/tri1/final.mdl
occs=exp/tri1/final.occs
tree=exp/tri1/tree
graphdir=exp/graph_tri1
silphones=`cat data/silphones.csl`

regtree=$dir/regtree
maxleaves=2     # max # of regression-tree leaves.
fmllr_mincount=5000   # mincount before we add new transform.
mllr_mincount=1000    # mincount before we add new transform.
gmm-make-regtree --sil-phones=$silphones --state-occs=$occs \
  --max-leaves=$maxleaves $model $regtree 2>$dir/make_regtree.out

scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  # Comment the two lines below to make this per-utterance.
  spk2utt_opt=--spk2utt=ark:data/test_${test}.spk2utt
  utt2spk_opt=--utt2spk=ark:data/test_${test}.utt2spk

  feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- |"

  ( ali-to-post ark:$srcdir/test_${test}.ali ark:- | \
    weight-silence-post 0.01 $silphones $model ark:- ark:- | \
    gmm-est-fmllr --fmllr-min-count=$fmllr_mincount $spk2utt_opt $model \
     "$feats" ark,o:- ark:$dir/${test}.fmllr ) 2>$dir/fmllr_${test}.log

  adapt_feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- | transform-feats $utt2spk_opt ark:$dir/${test}.fmllr ark:- ark:- |"

  gmm-decode-faster --beam=20.0 --acoustic-scale=0.08333 \
    --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst \
    "$adapt_feats" ark,t:$dir/${test}_pass2.tra ark,t:$dir/${test}_pass2.ali \
    2> $dir/pass2_${test}.log

  ( ali-to-post ark:$dir/${test}_pass2.ali ark:- | \
    gmm-est-regtree-mllr --mllr-min-count=$mllr_mincount $spk2utt_opt \
      $model "$adapt_feats" ark:- $regtree ark:$dir/${test}.mllr ) \
      2>$dir/mllr_${test}.log

  gmm-decode-faster-regtree-mllr $utt2spk_opt --acoustic-scale=0.08333 \
    --beam=20.0 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst \
    $regtree "$adapt_feats" ark:$dir/${test}.mllr ark,t:$dir/test_${test}.tra \
    ark,t:$dir/test_${test}.ali  2> $dir/decode_${test}.log

  # the ,p option lets it score partial output without dying..

  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
    compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra > $dir/wer_${test}
 ) &
done

wait

grep WER $dir/wer_* | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   > $dir/wer

