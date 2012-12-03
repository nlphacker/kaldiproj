#!/bin/bash

# Copyright 2012 Chao Weng 
# Apache 2.0

#exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

. cmd.sh


case 0 in    #goto here
    1)
;;           #here:
esac

# Data Preparation, 
local/hkust_data_prep.sh /mnt/spdb/LDC2005S15
# Lexicon Preparation,
local/hkust_prepare_dict.sh




# Phone Sets, questions, L compilation 
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

# LM training
local/hkust_train_lms.sh

# G compilation, check LG composition
local/hkust_format_data.sh

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in train dev; do 
  steps/make_mfcc.sh --nj 10 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done
# after this, the next command will remove the small number of utterances
# that couldn't be extracted for some reason (e.g. too short; no such file).
utils/fix_data_dir.sh data/train || exit 1;

steps/train_mono.sh --nj 10 \
  data/train data/lang exp/mono0a || exit 1;


# Monophone decoding
utils/mkgraph.sh --mono data/lang_test exp/mono0a exp/mono0a/graph || exit 1
# note: local/decode.sh calls the command line once for each
# test, and afterwards averages the WERs into (in this case
# exp/mono/decode/



steps/decode.sh --config conf/decode.config --nj 10 \
  exp/mono0a/graph data/dev exp/mono0a/decode



# Get alignments from monophone system.
steps/align_si.sh --nj 10 \
  data/train data/lang exp/mono0a exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh \
 2500 20000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

# decode tri1
utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
steps/decode.sh --config conf/decode.config --nj 10 \
  exp/tri1/graph data/dev exp/tri1/decode



# align tri1
steps/align_si.sh --nj 10 \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# train tri2 [delta+delta-deltas]
steps/train_deltas.sh \
 2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

# decode tri2
utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
steps/decode.sh --config conf/decode.config --nj 10 \
  exp/tri2/graph data/dev exp/tri2/decode

# train and decode tri2b [LDA+MLLT]

steps/align_si.sh --nj 10 \
  data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, 
steps/train_lda_mllt.sh \
 2500 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
steps/decode.sh --nj 10 --config conf/decode.config \
  exp/tri3a/graph data/dev exp/tri3a/decode
# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/align_fmllr.sh --nj 10 \
  data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh \
  2500 20000 data/train data/lang exp/tri3a_ali exp/tri4a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
steps/decode_fmllr.sh --nj 10 --config conf/decode.config \
  exp/tri4a/graph data/dev exp/tri4a/decode

steps/align_fmllr.sh --nj 10 \
  data/train data/lang exp/tri4a exp/tri4a_ali

# Building a larger SAT system.

steps/train_sat.sh \
  3500 100000 data/train data/lang exp/tri4a_ali exp/tri5a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
steps/decode_fmllr.sh --nj 10 --config conf/decode.config \
   exp/tri5a/graph data/dev exp/tri5a/decode || exit 1;


# MMI starting from system in tri5a.  Use the same data (100k_nodup).
# Later we'll use all of it.
steps/align_fmllr.sh --nj 10 \
  data/train data/lang exp/tri5a exp/tri5a_ali || exit 1;
steps/make_denlats.sh --nj 10 --transform-dir exp/tri5a_ali \
  --config conf/decode.config \
  data/train data/lang exp/tri5a exp/tri5a_denlats || exit 1;
steps/train_mmi.sh --boost 0.1 \
  data/train data/lang exp/tri5a_ali exp/tri5a_denlats exp/tri5a_mmi_b0.1 || exit 1;
steps/decode.sh --nj 10 --config conf/decode.config \
  --transform-dir exp/tri5a/decode \
  exp/tri5a/graph data/dev exp/tri5a_mmi_b0.1/decode || exit 1 ; 

# Do MPE.
steps/train_mpe.sh data/train data/lang exp/tri5a_ali exp/tri5a_denlats exp/tri5a_mpe || exit 1;

steps/decode.sh --nj 10 --config conf/decode.config \
  --transform-dir exp/tri5a/decode \
  exp/tri5a/graph data/dev exp/tri5a_mpe/decode || exit 1 ;
# Do MCE.

steps/train_mce.sh data/train data/lang exp/tri5a_ali exp/tri5a_denlats exp/tri5a_mce || exit 1;

steps/decode.sh --nj 10 --config conf/decode.config \
  --transform-dir exp/tri5a/decode \
  exp/tri5a/graph data/dev exp/tri5a_mce/decode || exit 1 ;

# getting results (see RESULTS file)
for x in exp/*/decode; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null
for x in exp/*/decode; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

exit 1;

