#!/bin/bash


exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.


# Data prep

local/swbd_p1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2

local/swbd_p1_train_lms.sh

local/swbd_p1_format_data.sh

# Data preparation and formatting for eval2000 (note: the "text" file
# is not very much preprocessed; for actual WER reporting we'll use
# sclite.
local/eval2000_data_prep.sh /mnt/matylda2/data/HUB5_2000/ /mnt/matylda2/data/HUB5_2000/

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
#mfccdir=/mnt/matylda6/ijanda/kaldi_swbd_mfcc
mfccdir=/mnt/matylda6/jhu09/qpovey/kaldi_swbd_mfcc
cmd="queue.pl -q all.q@@blade" # remove the option if no queue.
local/make_mfcc_segs.sh --num-jobs 10 --cmd "$cmd" data/train exp/make_mfcc/train $mfccdir
# after this, the next command will remove the small number of utterances
# that couldn't be extracted for some reason (e.g. too short; no such file).
scripts/fix_data_dir.sh data/train

local/make_mfcc_segs.sh --num-jobs 4 data/eval2000 exp/make_mfcc/eval2000 $mfccdir
scripts/fix_data_dir.sh data/eval2000 # remove segments that had problems, e.g. too short.

# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.

scripts/subset_data_dir.sh --first data/train 4000 data/train_dev # 5.3 hours.
n=$[`cat data/train/segments | wc -l` - 4000]
scripts/subset_data_dir.sh --last data/train $n data/train_nodev


# Now-- there are 264k utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.
scripts/subset_data_dir.sh --shortest data/train_nodev 100000 data/train_100kshort
scripts/subset_data_dir.sh  data/train_100kshort 10000 data/train_10k
local/remove_dup_utts.sh 100 data/train_10k data/train_10k_nodup

# Take the first 30k utterances (about 1/8th of the data)
scripts/subset_data_dir.sh --first data/train_nodev 30000 data/train_30k
local/remove_dup_utts.sh 200 data/train_30k data/train_30k_nodup

local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup

# Take the first 100k utterances (just under half the data); we'll use
# this for later stages of training.
scripts/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup


( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_10k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_10k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_swbd_10k_nodup.ark,$mfccdir/kaldi_swbd_10k_nodup.scp \
  && cp $mfccdir/kaldi_swbd_10k_nodup.scp data/train_10k_nodup/feats.scp
)

( . path.sh; 
  # make sure mfccdir is defined as above..
  cp data/train_30k_nodup/feats.scp{,.bak} 
  copy-feats scp:data/train_30k_nodup/feats.scp  ark,scp:$mfccdir/kaldi_swbd_30k_nodup.ark,$mfccdir/kaldi_swbd_30k_nodup.scp \
  && cp $mfccdir/kaldi_swbd_30k_nodup.scp data/train_30k_nodup/feats.scp
)
 


decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
long_cmd="queue.pl -q long.q@@blade -l ram_free=700M,mem_free=700M"

decode_opts1="--beam 11.0" # for one-pass decoding
decode_opts2="--beam1 8.0 --beam2 11.0 " # for two-pass decoding.


steps/train_mono.sh --num-jobs 10 --cmd "$train_cmd" \
  data/train_10k_nodup data/lang exp/mono0a

steps/align_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
    2500 20000 data/train_30k_nodup data/lang exp/mono0a_ali exp/tri1

steps/align_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/tri1 exp/tri1_ali

steps/train_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
    2500 20000 data/train_30k_nodup data/lang exp/tri1_ali exp/tri2

steps/align_deltas.sh --num-jobs 30 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/tri2 exp/tri2_ali

# Train tri3a, which is LDA+MLLT, on 30k_nodup data.
steps/train_lda_mllt.sh --num-jobs 30 --cmd "$train_cmd" \
   2500 20000 data/train_30k_nodup data/lang exp/tri2_ali exp/tri3a

# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/align_lda_mllt_sat.sh --num-jobs 30 --cmd "$train_cmd" \
 data/train_100k_nodup data/lang exp/tri3a exp/tri3a_ali_100k_nodup



steps/train_lda_mllt_sat.sh  --num-jobs 30 --cmd "$train_cmd" \
  2500 20000 data/train_100k_nodup data/lang exp/tri3a_ali_100k_nodup exp/tri4a

scripts/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
scripts/decode.sh -l data/lang_test --num-jobs 30 --cmd "$decode_cmd" --opts "$decode_opts2" \
 steps/decode_lda_mllt_sat.sh exp/tri4a/graph data/eval2000 exp/tri4a/decode_eval2000

scripts/decode.sh  --num-jobs 30 --cmd "$decode_cmd" --opts "$decode_opts2" \
 steps/decode_lda_mllt_sat.sh exp/tri4a/graph data/train_dev exp/tri4a/decode_train_dev

steps/align_lda_mllt_sat.sh --num-jobs 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri4a exp/tri4a_ali_100k_nodup


( # Build a SGMM system on just the 100k_nodup data, on top of LDA+MLLT+SAT.
 steps/train_ubm_lda_etc.sh --num-jobs 30 --cmd "$train_cmd" \
   700 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/ubm5a
 steps/train_sgmm_lda_etc.sh --num-jobs 30 --cmd "$train_cmd" \
   4500 40000 50 40 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup \
     exp/ubm5a/final.ubm exp/sgmm5a

 scripts/mkgraph.sh data/lang_test exp/sgmm5a exp/sgmm5a/graph
 scripts/decode.sh --opts "$decode_opts1" -l data/lang_test --num-jobs 30 --cmd "$decode_cmd" \
   steps/decode_sgmm_lda_etc.sh exp/sgmm5a/graph data/eval2000 exp/sgmm5a/decode_eval2000 \
   exp/tri4a/decode_eval2000
 scripts/decode.sh --opts "$decode_opts1" --num-jobs 30 --cmd "$decode_cmd" \
   steps/decode_sgmm_lda_etc.sh exp/sgmm5a/graph data/train_dev exp/sgmm5a/decode_train_dev \
   exp/tri4a/decode_train_dev


  # This decoding script doesn't do a full re-decoding but is limited to the lattices
  # from the baseline decoding.
  scripts/decode.sh -l data/lang_test --num-jobs 30 --cmd "$decode_cmd" steps/decode_sgmm_lda_etc_fromlats.sh \
    data/lang_test data/eval2000 exp/sgmm5a/decode_eval2000_fromlats exp/tri4a/decode_eval2000


)



# note: was 4k,150k when I was using all the data, may change back.
steps/train_lda_mllt_sat.sh  --num-jobs 30 --cmd "$train_cmd" \
  3500 75000 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/tri5a

scripts/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph

scripts/decode.sh --opts "$decode_opts2" \
  -l data/lang_test --num-jobs 30 --cmd "$decode_cmd" \
  steps/decode_lda_mllt_sat.sh exp/tri5a/graph data/eval2000 exp/tri5a/decode_eval2000

scripts/decode.sh --opts "$decode_opts2" \
  --num-jobs 30 --cmd "$decode_cmd" \
  steps/decode_lda_mllt_sat.sh exp/tri5a/graph data/train_dev exp/tri5a/decode_train_dev



( # Try mixing up from the 5a system to see if more Gaussians helps.
  steps/align_lda_mllt_sat.sh --num-jobs 30 --cmd "$train_cmd" \
    data/train_100k_nodup data/lang exp/tri5a exp/tri5a_ali_100k_nodup
 steps/mixup_lda_etc.sh --num-jobs 30 --cmd "$train_cmd" \
  100000 data/train_100k_nodup exp/tri5a exp/tri5a_ali_100k_nodup exp/tri5a_100k
 scripts/decode.sh --opts "$decode_opts2" \
    --num-jobs 30 --cmd "$decode_cmd" \
   steps/decode_lda_mllt_sat.sh exp/tri5a/graph data/train_dev exp/tri5a_100k/decode_train_dev
)


( 
  # MMI starting from the system in tri5a.
  steps/align_lda_mllt_sat.sh --num-jobs 40 --cmd "$train_cmd" \
    data/train_100k data/lang exp/tri5a exp/tri5a_ali_100k
  # Use a smaller beam for Switchboard, as in test time.  Use the 100k dataset,
  # but include duplicates for discriminative training.
  steps/make_denlats_lda_etc.sh --num-jobs 40 --sub-split 40 --cmd "$train_cmd" \
    $decode_opts1 --lattice-beam 6.0 data/train_100k data/lang exp/tri5a_ali_100k exp/tri5a_denlats_100k
  steps/train_lda_etc_mmi.sh --num-jobs 40 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri5a_ali_100k exp/tri5a_denlats_100k exp/tri5a exp/tri5a_mmi
  scripts/decode.sh -l data/lang_test --num-jobs 30 --cmd "$decode_cmd" --opts "$decode_opts1" \
     steps/decode_lda_etc.sh exp/tri5a/graph data/eval2000 exp/tri5a_mmi/decode_eval2000 \
     exp/tri5a/decode_eval2000
  steps/train_lda_etc_mmi.sh --boost 0.1 --num-jobs 40 --cmd "$train_cmd" \
    data/train_100k data/lang exp/tri5a_ali_100k exp/tri5a_denlats_100k \
    exp/tri5a exp/tri5a_mmi_b0.1
  scripts/decode.sh -l data/lang_test --num-jobs 30 --cmd "$decode_cmd" --opts "$decode_opts1" \
    steps/decode_lda_etc.sh exp/tri5a/graph  data/eval2000 exp/tri5a_mmi_b0.1/decode_eval2000 \
    exp/tri5a/decode_eval2000
)


# Align the 5a system; we'll train triphone and SGMM systems on
# all the data, on top of this.
steps/align_lda_mllt_sat.sh  --num-jobs 30 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri5a exp/tri5a_ali_nodup

( # Train triphone system on all the data.
 steps/train_lda_mllt_sat.sh  --num-jobs 30 --cmd "$train_cmd" \
   4000 150000 data/train_nodup data/lang exp/tri5a_ali_nodup exp/tri6a

 scripts/mkgraph.sh data/lang_test exp/tri6a exp/tri6a/graph
 scripts/decode.sh --opts "$decode_opts2" \
   -l data/lang_test --num-jobs 30 --cmd "$decode_cmd" \
   steps/decode_lda_mllt_sat.sh exp/tri6a/graph data/eval2000 exp/tri6a/decode_eval2000

 scripts/decode.sh --opts "$decode_opts2" \
   --num-jobs 30 --cmd "$decode_cmd" \
   steps/decode_lda_mllt_sat.sh exp/tri6a/graph data/train_dev exp/tri6a/decode_train_dev

)



# getting results (see RESULTS file)
for x in exp/*/decode_*; do [ -d $x ] && grep Sum $x/score_*/*.sys | scripts/best_wer.sh; done 2>/dev/null
for x in exp/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | scripts/best_wer.sh; done 2>/dev/null


