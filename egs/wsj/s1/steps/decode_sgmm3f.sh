#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

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


# This script does the decoding of a single batch of test data (on one core).
# It requires arguments.  It takes the graphdir and decoding directory,
# and the job number which can actually be any string (even ""); it expects
# a file $decode_dir/test${job_number}.scp to exist, and puts its output in
# $decode_dir/${job_number}.tra


if [ $# != 4 ]; then
   echo "Usage: steps/decode_sgmm3f.sh <graph> <decode-dir> <job-number> <old-graph>"
   exit 1;
fi

. path.sh || exit 1;

oldacwt=0.0625
acwt=0.0769 # 1/13


prebeam=12.0
beam=13.0
max_active=7000
silphones=`cat data/silphones.csl`
model=exp/sgmm3f/final.mdl
alimodel=exp/sgmm3f/final.alimdl
oldmodel=exp/tri2k/final.mdl
oldalimodel=exp/tri2k/final.alimdl

ldamat=exp/tri2k/lda.mat
defaultmat=exp/tri2k/default.mat
et=exp/tri2k/final.et

preselectmap=exp/ubm3b/preselect.map
graph=$1
dir=$2
job=$3
oldgraph=$4
scp=$dir/$job.scp
defaultfeats="ark:splice-feats scp:$scp ark:- | transform-feats $defaultmat ark:- ark:- |"
sifeats="ark:splice-feats scp:$scp ark:- | transform-feats $ldamat ark:- ark:- |"

filenames="$scp $oldmodel $oldalimodel $ldamat $defaultmat $model $graph $oldgraph data/words.txt"

for file in $filenames; do
  if [ ! -f $file ] ; then
    echo "No such file $file";
    exit 1;
  fi
done

if [ -f $dir/$job.spk2utt ]; then
  if [ ! -f $dir/$job.utt2spk ]; then
     echo "spk2utt but not utt2spk file present!"
     exit 1
  fi
  spk2utt_opt=--spk2utt=ark:$dir/$job.spk2utt
  utt2spk_opt=--utt2spk=ark:$dir/$job.utt2spk
fi

# Do first-pass decoding with alignment model from tri2k (will get exponential transform
# from this).
 
gmm-decode-faster --beam=$prebeam --max-active=$max_active --acoustic-scale=$oldacwt --word-symbol-table=data/words.txt $oldalimodel $oldgraph "$defaultfeats" ark,t:$dir/$job.pre1_tra ark,t:$dir/$job.pre1_ali  2>>$dir/pre1decode${job}.log  || exit 1;
 
# Estimate transforms
(ali-to-post ark:$dir/$job.pre1_ali ark:- | \
  weight-silence-post 0.0 $silphones $oldalimodel ark:- ark:- | \
  gmm-post-to-gpost $oldalimodel "$defaultfeats" ark,o:- ark:- | \
  gmm-est-et --normalize-type=diag $spk2utt_opt $oldmodel $et "$sifeats" ark,o:- \
     ark:$dir/$job.et_trans ark,t:$dir/$job.warp ) 2>$dir/et${job}.log || exit 1;
 
feats="ark:splice-feats --print-args=false scp:$scp ark:- | transform-feats $ldamat ark:- ark:- | transform-feats $utt2spk_opt ark:$dir/$job.et_trans ark:- ark:- |"


cat data/eval*.utt2spk | \
 scripts/compose_maps.pl - data/spk2gender.map | \
 scripts/compose_maps.pl - $preselectmap | \
 scripts/filter_scp.pl $scp - | \
   gzip -c > $dir/preselect.$job.gz


echo running on `hostname` > $dir/decode${job}.log

sgmm-gselect "--preselect=ark:gunzip -c $dir/preselect.$job.gz|" \
     $model "$feats" ark,t:- 2>$dir/gselect${job}.log | \
     gzip -c > $dir/gselect${job}.gz || exit 1;
gselect_opt="--gselect=ark:gunzip -c $dir/gselect${job}.gz|"


sgmm-decode-faster "$gselect_opt" --beam=$prebeam --max-active=$max_active \
   --acoustic-scale=$acwt \
   --word-symbol-table=data/words.txt $alimodel $graph "$feats" \
   ark,t:$dir/$job.pre2_tra ark,t:$dir/$job.pre2_ali  2>$dir/predecode${job}.log  || exit 1;

( ali-to-post ark:$dir/${job}.pre2_ali ark:- | \
  weight-silence-post 0.01 $silphones $alimodel ark:- ark:- | \
  sgmm-post-to-gpost "$gselect_opt" $alimodel "$feats" ark,s,cs:- ark:- | \
  sgmm-est-spkvecs-gpost $spk2utt_opt $model "$feats" ark,s,cs:- \
     ark,t:$dir/${job}.vecs1 ) 2>$dir/vecs1.${job}.log || exit 1;

( ali-to-post ark:$dir/${job}.pre2_ali ark:- | \
  weight-silence-post 0.01 $silphones $alimodel ark:- ark:- | \
  sgmm-est-spkvecs "$gselect_opt" --spk-vecs=ark,t:$dir/${job}.vecs1 $spk2utt_opt $model \
   "$feats" ark,s,cs:- ark,t:$dir/${job}.vecs2 ) 2>$dir/vecs2.${job}.log || exit 1;

sgmm-decode-faster "$gselect_opt" --beam=$beam --max-active=$max_active \
   $utt2spk_opt --spk-vecs=ark:$dir/${job}.vecs2 \
   --acoustic-scale=$acwt \
   --word-symbol-table=data/words.txt $model $graph "$feats" \
   ark,t:$dir/$job.tra ark,t:$dir/$job.ali  2>$dir/decode${job}.log  || exit 1;


