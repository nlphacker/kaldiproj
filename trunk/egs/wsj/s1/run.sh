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

exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# might want to run this script on a machine that has plenty of memory.

# (1) To get the CMU dictionary, do:
svn co https://cmusphinx.svn.sourceforge.net/svnroot/cmusphinx/trunk/cmudict/
# got this at revision 10742 in my current test.  can add -r 10742 for strict
# compatibility.

#(2) Dictionary preparation:

mkdir -p data

# Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
# We are adding suffixes _B, _E, _S for beginning, ending, and singleton phones.

cat cmudict/cmudict.0.7a.symbols | perl -ane 's:\r::; print;' | \
 awk 'BEGIN{print "<eps> 0"; print "SIL 1"; print "SPN 2"; print "NSN 3"; N=4; } 
           {printf("%s %d\n", $1, N++); }
           {printf("%s_B %d\n", $1, N++); }
           {printf("%s_E %d\n", $1, N++); }
           {printf("%s_S %d\n", $1, N++); } ' >data/phones.txt


# First make a version of the lexicon without the silences etc, but with the position-markers.
# Remove the comments from the cmu lexicon and remove the (1), (2) from words with multiple 
# pronunciations.

grep -v ';;;' cmudict/cmudict.0.7a | perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' \
 | perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die;
   if(@A==1) { print "$w $A[0]_S\n"; } else { print "$w $A[0]_B ";
     for($n=1;$n<@A-1;$n++) { print "$A[$n] "; } print "$A[$n]_E\n"; } ' \
  > data/lexicon_nosil.txt

# Add to cmudict the silences, noises etc.

(echo '!SIL SIL'; echo '<s> '; echo '</s> '; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; echo '<NOISE> NSN'; ) | \
 cat - data/lexicon_nosil.txt  > data/lexicon.txt


silphones="SIL SPN NSN";
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/phones.txt "$silphones" data/silphones.csl data/nonsilphones.csl

# This adds disambig symbols to the lexicon and produces data/lexicon_disambig.txt

ndisambig=`scripts/add_lex_disambig.pl data/lexicon.txt data/lexicon_disambig.txt`
echo $ndisambig > data/lex_ndisambig
# Next, create a phones.txt file that includes the disambig symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
scripts/add_disambig.pl --include-zero data/phones.txt $ndisambig > data/phones_disambig.txt

# Make the words symbol-table; add the disambiguation symbol #0 (we use this in place of epsilon
# in the grammar FST).
cat data/lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > data/words.txt


#(3)
# data preparation (this step requires the WSJ disks, from LDC).
# It takes as arguments a list of the directories ending in
# e.g. 11-13.1 (we don't assume a single root dir because
# there are different ways of unpacking them).

cd data_prep


# On BUT system, do:
# The following command needs a list of directory names from
# the LDC's WSJ disks.  These will end in e.g. 11-1.1.
# examples:
# /ais/gobi2/speech/WSJ/*/??-{?,??}.?
# /mnt/matylda2/data/WSJ?/??-{?,??}.?
./run.sh [list-of-directory-names]


cd ..



# Here is where we select what data to train on.
# use all the si284 data.
cp data_prep/train_si284_wav.scp data/train_wav.scp
cp data_prep/train_si284.txt data/train.txt
cp data_prep/train_si284.spk2utt data/train.spk2utt 
cp data_prep/train_si284.utt2spk data/train.utt2spk
cp data_prep/spk2gender.map data/

for x in eval_nov92 dev_nov93 eval_nov93; do 
  cp data_prep/$x.spk2utt data/$x.spk2utt
  cp data_prep/$x.utt2spk data/$x.utt2spk
  cp data_prep/$x.txt data/$x.txt
done

for x in train eval_nov92 dev_nov93 eval_nov93; do  
 cat data/$x.txt | scripts/sym2int.pl --ignore-first-field data/words.txt  > data/$x.tra
done


# Get the right paths on our system by sourcing the following shell file
# (edit it if it's not right for your setup). 
. path.sh

# Create the basic L.fst without disambiguation symbols, for use
# in training. 
scripts/make_lexicon_fst.pl data/lexicon.txt 0.5 SIL | \
  fstcompile --isymbols=data/phones.txt --osymbols=data/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/L.fst

# Create the lexicon FST with disambiguation symbols.  There is an extra
# step where we create a loop "pass through" the disambiguation symbols
# from G.fst.  

phone_disambig_symbol=`grep \#0 data/phones_disambig.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 data/words.txt | awk '{print $2}'`

scripts/make_lexicon_fst.pl data/lexicon_disambig.txt 0.5 SIL  | \
   fstcompile --isymbols=data/phones_disambig.txt --osymbols=data/words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > data/L_disambig.fst


# Making the grammar FSTs 
# This step is quite specific to this WSJ setup.
# see data_prep/run.sh for more about where these LMs came from.

steps/make_lm_fsts.sh

## Sanity check; just making sure the next command does not crash. 
fstdeterminizestar data/G_bg.fst >/dev/null  

## Sanity check; just making sure the next command does not crash. 
fsttablecompose data/L_disambig.fst data/G_bg.fst | fstdeterminizestar >/dev/null


# At this point, make sure that "./exp/" is somewhere you can write
# a reasonably large amount of data (i.e. on a fast and large 
# disk somewhere).  It can be a soft link if necessary.


# (4) feature generation


# Make the training features.
# note that this runs 3-4 times faster if you compile with DEBUGLEVEL=0
# (this turns on optimization).

# Set "dir" to someplace you can write to.
dir=/mnt/matylda6/jhu09/qpovey/kaldi_wsj2_mfcc_e
steps/make_mfcc_train.sh $dir
steps/make_mfcc_test.sh $dir


# (5) running the training and testing steps..

steps/train_mono.sh || exit 1;

(scripts/mkgraph.sh --mono data/G_tg_pruned.fst exp/mono/tree exp/mono/final.mdl exp/graph_mono_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_mono_tgpr_eval92 exp/graph_mono_tg_pruned/HCLG.fst steps/decode_mono.sh data/eval_nov92.scp ) &

steps/train_tri1.sh || exit 1;

# add --no-queue --num-jobs 4 after "scripts/decode.sh" below, if you don't have
# qsub on your system.  The number of jobs to use depends on how many CPUs and
# how much memory you have, on the local machine.  If you do have qsub on your
# system, you will probably have to edit steps/decode.sh anyway to change the
# queue options... or if you have a different queueing system, you'd have to
# modify the script to use that.
 
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri1/tree exp/tri1/final.mdl exp/graph_tri1_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri1_tgpr_eval92 exp/graph_tri1_tg_pruned/HCLG.fst steps/decode_tri1.sh data/eval_nov92.scp ) &

steps/train_tri2a.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2a/tree exp/tri2a/final.mdl exp/graph_tri2a_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2a_tgpr_eval92 exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a.sh data/eval_nov92.scp 
 scripts/decode.sh exp/decode_tri2a_tgpr_eval93 exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a.sh data/eval_nov93.scp 

 scripts/decode.sh exp/decode_tri2a_tgpr_fmllr_utt_eval92 exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_fmllr.sh data/eval_nov92.scp 
 scripts/decode.sh --per-spk exp/decode_tri2a_tgpr_fmllr_eval92 exp/graph_tri2a_tg_pruned/HCLG.fst steps/decode_tri2a_fmllr.sh data/eval_nov92.scp 


) &


steps/train_tri3a.sh || exit 1;

(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri3a/tree exp/tri3a/final.mdl exp/graph_tri3a_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri3a_tgpr_eval92 exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a.sh data/eval_nov92.scp 
# per-speaker fMLLR
scripts/decode.sh --per-spk exp/decode_tri3a_tgpr_fmllr_eval92 exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_fmllr.sh data/eval_nov92.scp
# per-utterance fMLLR
scripts/decode.sh exp/decode_tri3a_tgpr_uttfmllr_eval92 exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_fmllr.sh data/eval_nov92.scp 
# per-speaker diagonal fMLLR
scripts/decode.sh --per-spk exp/decode_tri3a_tgpr_dfmllr_eval92 exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_diag_fmllr.sh data/eval_nov92.scp 
# per-utterance diagonal fMLLR
scripts/decode.sh exp/decode_tri3a_tgpr_uttdfmllr_eval92 exp/graph_tri3a_tg_pruned/HCLG.fst steps/decode_tri3a_diag_fmllr.sh data/eval_nov92.scp 
)&

# will delete:
## scripts/decode_queue_fmllr.sh exp/graph_tri3a_tg_pruned exp/tri3a/final.mdl exp/decode_tri3a_tg_pruned_fmllr &

#### Now alternative experiments... ###

# ET
steps/train_tri2b.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2b/tree exp/tri2b/final.mdl exp/graph_tri2b_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2b_tgpr_utt_eval92 exp/graph_tri2b_tg_pruned/HCLG.fst steps/decode_tri2b.sh data/eval_nov92.scp 
 scripts/decode.sh --per-spk exp/decode_tri2b_tgpr_eval92 exp/graph_tri2b_tg_pruned/HCLG.fst steps/decode_tri2b.sh data/eval_nov92.scp 
 scripts/decode.sh exp/decode_tri2b_tgpr_utt_fmllr_eval92 exp/graph_tri2b_tg_pruned/HCLG.fst steps/decode_tri2b_fmllr.sh data/eval_nov92.scp 
 scripts/decode.sh --per-spk exp/decode_tri2b_tgpr_fmllr_eval92 exp/graph_tri2b_tg_pruned/HCLG.fst steps/decode_tri2b_fmllr.sh data/eval_nov92.scp 
) &

# MLLT/STC
steps/train_tri2d.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2d/tree exp/tri2d/final.mdl exp/graph_tri2d_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2d_tgpr_eval92 exp/graph_tri2d_tg_pruned/HCLG.fst steps/decode_tri2d.sh data/eval_nov92.scp  )&

# Splice+LDA
steps/train_tri2e.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2e/tree exp/tri2e/final.mdl exp/graph_tri2e_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2e_tgpr_eval92 exp/graph_tri2e_tg_pruned/HCLG.fst steps/decode_tri2e.sh data/eval_nov92.scp  )&

# Splice+LDA+MLLT
steps/train_tri2f.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2f/tree exp/tri2f/final.mdl exp/graph_tri2f_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2f_tgpr_eval92 exp/graph_tri2f_tg_pruned/HCLG.fst steps/decode_tri2f.sh data/eval_nov92.scp  )&

# Linear VTLN (+ regular VTLN)
steps/train_tri2g.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2g/tree exp/tri2g/final.mdl exp/graph_tri2g_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2g_tgpr_utt_eval92 exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g.sh data/eval_nov92.scp  
 scripts/decode.sh exp/decode_tri2g_tgpr_utt_diag_eval92 exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_diag.sh data/eval_nov92.scp  
 scripts/decode.sh --wav exp/decode_tri2g_tgpr_utt_vtln_diag_eval92 exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_vtln_diag.sh data/eval_nov92.scp  

 scripts/decode.sh --per-spk exp/decode_tri2g_tgpr_eval92 exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g.sh data/eval_nov92.scp  
 scripts/decode.sh --per-spk exp/decode_tri2g_tgpr_diag_eval92 exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_diag.sh data/eval_nov92.scp  
 scripts/decode.sh --wav --per-spk exp/decode_tri2g_tgpr_vtln_diag_eval92 exp/graph_tri2g_tg_pruned/HCLG.fst steps/decode_tri2g_vtln_diag.sh data/eval_nov92.scp  

)&

# Splice+HLDA
steps/train_tri2h.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2h/tree exp/tri2h/final.mdl exp/graph_tri2h_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2h_tgpr_eval92 exp/graph_tri2h_tg_pruned/HCLG.fst steps/decode_tri2h.sh data/eval_nov92.scp  )&

# Triple-deltas + HLDA
steps/train_tri2i.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2i/tree exp/tri2i/final.mdl exp/graph_tri2i_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2i_tgpr_eval92 exp/graph_tri2i_tg_pruned/HCLG.fst steps/decode_tri2i.sh data/eval_nov92.scp  )&

# Splice + HLDA
steps/train_tri2j.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2j/tree exp/tri2j/final.mdl exp/graph_tri2j_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2j_tgpr_eval92 exp/graph_tri2j_tg_pruned/HCLG.fst steps/decode_tri2j.sh data/eval_nov92.scp  )&


# LDA+ET
steps/train_tri2k.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2k/tree exp/tri2k/final.mdl exp/graph_tri2k_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2k_tgpr_utt_eval92 exp/graph_tri2k_tg_pruned/HCLG.fst steps/decode_tri2k.sh data/eval_nov92.scp 
 scripts/decode.sh --per-spk exp/decode_tri2k_tgpr_eval92 exp/graph_tri2k_tg_pruned/HCLG.fst steps/decode_tri2k.sh data/eval_nov92.scp 
 )&

# LDA+MLLT+SAT
steps/train_tri2l.sh
(scripts/mkgraph.sh data/G_tg_pruned.fst exp/tri2l/tree exp/tri2l/final.mdl exp/graph_tri2l_tg_pruned || exit 1;
 scripts/decode.sh exp/decode_tri2l_tgpr_utt_eval92 exp/graph_tri2l_tg_pruned/HCLG.fst steps/decode_tri2l.sh data/eval_nov92.scp 
 scripts/decode.sh --per-spk exp/decode_tri2l_tgpr_eval92 exp/graph_tri2l_tg_pruned/HCLG.fst steps/decode_tri2l.sh data/eval_nov92.scp 
 )&





# Note on WERs at different stages of decoding:
#exp/decode_mono_tg_pruned/wer:%WER 31.82 [ 1795 / 5641, 109 ins, 412 del, 1274 sub ]
#exp/decode_tri1_tg_pruned/wer:%WER 13.61 [ 768 / 5641, 134 ins, 76 del, 558 sub ]
#exp/decode_tri2a_tg_pruned/wer:%WER 12.94 [ 730 / 5641, 131 ins, 62 del, 537 sub ]
#exp/decode_tri3a_tg_pruned/wer:%WER 10.88 [ 614 / 5641, 126 ins, 47 del, 441 sub ]


# For an e.g. of scoring with sclite: do e.g.
#  scripts/score_sclite.sh exp/decode_tri2a_tg_pruned 
