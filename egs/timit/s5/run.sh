#!/bin/bash

# Copyright 2013  Bagher BabaAli

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh

# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

decode_nj=20
train_nj=30

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT

local/timit_data_prep.sh $timit || exit 1;

local/timit_prepare_dict.sh || exit 1;

utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "sil" data/local/lang_tmp data/lang || exit 1;

local/timit_format_data.sh || exit 1;

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set           "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc
use_pitch=false
use_ffv=false

for x in train dev test; do 
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================

steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang exp/mono || exit 1;

utils/mkgraph.sh --mono data/lang_test_bg exp/mono exp/mono/graph || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/mono/graph data/dev exp/mono/decode_dev || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/mono/graph data/test exp/mono/decode_test || exit 1;

echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
 data/train data/lang exp/mono exp/mono_ali || exit 1;

# Train tri1, which is deltas + delta-deltas, on train data.
steps/train_deltas.sh --cmd "$train_cmd" \
 $numLeavesTri1 $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri1/graph data/dev exp/tri1/decode_dev || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri1/graph data/test exp/tri1/decode_test || exit 1;

echo ============================================================================
echo "                 tri2 : LDA + MLLT Training & Decoding                    "
echo ============================================================================

steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
 --splice-opts "--left-context=3 --right-context=3" \
 $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri2/graph data/dev exp/tri2/decode_dev || exit 1;

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri2/graph data/test exp/tri2/decode_test || exit 1;

echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
echo ============================================================================

# Align tri2 system with train data.
steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
 --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

# From tri2 system, train tri3 which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
 $numLeavesSAT $numGaussSAT data/train data/lang exp/tri2_ali exp/tri3 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph || exit 1;

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri3/graph data/dev exp/tri3/decode_dev || exit 1;

steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/tri3/graph data/test exp/tri3/decode_test || exit 1;

echo ============================================================================
echo "                        SGMM2 Training & Decoding                         "
echo ============================================================================

steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 data/train data/lang exp/tri3 exp/tri3_ali || exit 1;

steps/train_ubm.sh --cmd "$train_cmd" \
 $numGaussUBM data/train data/lang exp/tri3_ali exp/ubm4 || exit 1;

steps/train_sgmm2.sh --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
 data/train data/lang exp/tri3_ali exp/ubm4/final.ubm exp/sgmm2_4 || exit 1;

utils/mkgraph.sh data/lang_test_bg exp/sgmm2_4 exp/sgmm2_4/graph || exit 1;

steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
 --transform-dir exp/tri3/decode_dev exp/sgmm2_4/graph data/dev \
 exp/sgmm2_4/decode_dev || exit 1;

steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
 --transform-dir exp/tri3/decode_test exp/sgmm2_4/graph data/test \
 exp/sgmm2_4/decode_test || exit 1;

echo ============================================================================
echo "                    MMI + SGMM2 Training & Decoding                       "
echo ============================================================================

steps/align_sgmm2.sh --nj "$train_nj" --cmd "$train_cmd" \
 --transform-dir exp/tri3_ali --use-graphs true --use-gselect true data/train \
 data/lang exp/sgmm2_4 exp/sgmm2_4_ali || exit 1;

steps/make_denlats_sgmm2.sh --nj "$train_nj" --sub-split "$train_nj" --cmd "$decode_cmd"\
 --transform-dir exp/tri3_ali data/train data/lang exp/sgmm2_4_ali \
 exp/sgmm2_4_denlats || exit 1;

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" \
 --transform-dir exp/tri3_ali --boost 0.1 --zero-if-disjoint true \
 data/train data/lang exp/sgmm2_4_ali exp/sgmm2_4_denlats \
 exp/sgmm2_4_mmi_b0.1 || exit 1;

for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3/decode_dev data/lang_test_bg data/dev \
   exp/sgmm2_4/decode_dev exp/sgmm2_4_mmi_b0.1/decode_dev_it$iter || exit 1;

  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri3/decode_test data/lang_test_bg data/test \
   exp/sgmm2_4/decode_test exp/sgmm2_4_mmi_b0.1/decode_test_it$iter || exit 1;
done

echo ============================================================================
echo "                    DNN Hybrid Training & Decoding                        "
echo ============================================================================

# DNN hybrid system training parameters
dnn_mem_reqs="mem_free=1.0G,ram_free=0.2G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

steps/train_nnet_cpu.sh --mix-up 5000 --initial-learning-rate 0.015 \
  --final-learning-rate 0.002 --num-hidden-layers 2 --num-parameters 1500000 \
  --num-jobs-nnet "$train_nj" --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" \
  data/train data/lang exp/tri3_ali exp/tri4_nnet  || exit 1;

decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
  --transform-dir exp/tri3/decode_dev exp/tri3/graph data/dev \
  exp/tri4_nnet/decode_dev | tee exp/tri4_nnet/decode_dev/decode.log

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
  --transform-dir exp/tri3/decode_test exp/tri3/graph data/test \
  exp/tri4_nnet/decode_test | tee exp/tri4_nnet/decode_test/decode.log

echo ============================================================================
echo "                    System Combination (DNN+SGMM)                         "
echo ============================================================================

for iter in 1 2 3 4; do
  local/score_combine.sh --cmd "$decode_cmd" \
   data/dev data/lang_test_bg exp/tri4_nnet/decode_dev \
   exp/sgmm2_4_mmi_b0.1/decode_dev_it$iter exp/combine_2/decode_dev_it$iter

  local/score_combine.sh --cmd "$decode_cmd" \
   data/test data/lang_test_bg exp/tri4_nnet/decode_test \
   exp/sgmm2_4_mmi_b0.1/decode_test_it$iter exp/combine_2/decode_test_it$iter
done


echo ============================================================================
echo "                    Getting Results [see RESULTS file]                    "
echo ============================================================================

for x in exp/*/decode*; do
  [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh
done 

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0