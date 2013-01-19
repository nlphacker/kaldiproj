#!/bin/bash


# System and data directories
SCRIPT=$(readlink -f $0)
SysDir=`dirname $SCRIPT`


train_data_dir=/export/babel/oguz/10Hsubsets/106B-delivery-v0.2f_10hSubset/conversational/training
dev_data_dir=/export/babel/oguz/10Hsubsets/106B-delivery-v0.2f_10hSubset/conversational/dev_UnionOfDisjoints
lexicon_file=/export/a09/jtrmal/babel/egs/Tagalog-10hSystem2a/Lexicon/lexicon.txt


# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="--oov <unk>"

# Scoring protocols (dummy GLM file to appease the scoring script)
glmFile=`readlink -f ./conf/glm`

. ./local/CHECKPOINT.sh

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source the path.
. parse_options.sh || exit 1;

configfile=$1
[ -f $configfile ] && . $configfile
[ -f ./local.conf ] && . ./local.conf

echo ---------------------------------------------------------------
echo "Preparing lexicon in data/local on" `date`
echo ---------------------------------------------------------------
mkdir -p data/local
local/prepare_lexicon.pl \
    $lexiconFlags $lexicon_file data/local || exit 1

echo ---------------------------------------------------------------
echo "Creating L.fst etc in data/lang on" `date`
echo ---------------------------------------------------------------
mkdir -p data/lang
utils/prepare_lang.sh \
    --share-silence-phones true \
    data/local $oovSymbol data/local/tmp.lang data/lang || exit 1

echo -------------------------------------------------------------------------------
echo "Preparing acoustic training lists in data/train on" `date`
echo -------------------------------------------------------------------------------
mkdir -p data/train
local/prepare_acoustic_training_data.pl \
    --vocab data/local/lexicon.txt --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log || exit 1

echo --------------------------------------------------------------------
echo "Preparing dev data lists in data/dev on" `date`
echo --------------------------------------------------------------------
mkdir -p data/dev
local/prepare_acoustic_training_data.pl \
    --fragmentMarkers \-\*\~ \
    $dev_data_dir data/dev > data/dev/skipped_utts.log || exit 1

echo -------------------------------------------------------------------
echo "Preparing dev stm files in data/dev on" `date`
echo -------------------------------------------------------------------
local/prepare_stm.pl --fragmentMarkers \-\*\~ data/dev || exit 1
cp $glmFile data/dev

echo -------------------------------------------------------------------
echo "Creating a basic G.fst in data/lang on" `date`
echo -------------------------------------------------------------------
# Assume that the directory sriLM is already present and contains run.sh
#cd $SysDir/sriLM
local/train_lms_srilm.sh data data/srilm 
local/arpa2G.sh data/srilm/lm.gz data/lang data/lang

cd $SysDir
echo ----------------------------------------------------------------------
echo "Starting plp feature extraction in plp on" `date`
echo ----------------------------------------------------------------------
steps/make_plp.sh \
    --cmd "$train_cmd" --nj $nj_train \
    data/train exp/make_plp/train plp || exit 1
steps/compute_cmvn_stats.sh \
    data/train exp/make_plp/train plp || exit 1
# In case plp extraction failed on some utterance, delist them
utils/fix_data_dir.sh data/train

steps/make_plp.sh \
    --cmd "$train_cmd" --nj $nj_decode \
    data/dev exp/make_plp/dev plp || exit 1
steps/compute_cmvn_stats.sh \
    data/dev exp/make_plp/dev plp || exit 1
# In case plp extraction failed on some utterance, delist them
utils/fix_data_dir.sh data/dev
mkdir -p exp

echo -----------------------------------------------------------------------------------------
echo "Subsetting monophone training data in data/train_sub[123] on" `date`
echo -----------------------------------------------------------------------------------------
utils/subset_data_dir.sh data/train  5000 data/train_sub1 || exit 1
utils/subset_data_dir.sh data/train 10000 data/train_sub2 || exit 1
utils/subset_data_dir.sh data/train 20000 data/train_sub3 || exit 1

echo ---------------------------------------------------------------------------------
echo "Starting (small) monophone training in exp/mono on" `date`
echo ---------------------------------------------------------------------------------
steps/train_mono.sh \
    --boost-silence 1.5 --nj 24 --cmd "$train_cmd" \
    data/train_sub1 data/lang exp/mono || exit 1

echo ------------------------------------------------------------------------------
echo "Starting (small) triphone training in exp/tri1 on" `date`
echo ------------------------------------------------------------------------------
steps/align_si.sh \
    --boost-silence 1.5 --nj 24 --cmd "$train_cmd" \
    data/train_sub2 data/lang exp/mono exp/mono_ali_sub2 || exit 1
steps/train_deltas.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesTri1 $numGaussTri1 data/train_sub2 data/lang exp/mono_ali_sub2 exp/tri1 || exit 1

echo ----------------------------------------------------------------------------------------
echo "Spawning decoding with (small) triphone models in exp/tri1 on" `date`
echo ----------------------------------------------------------------------------------------
(
    mkdir -p exp/tri0/graph
    utils/mkgraph.sh data/lang exp/tri2 exp/tri0/graph &> exp/tri1/mkgraph.log
    mkdir -p exp/tri1/decode
    steps/decode.sh --nj $nj_decode --cmd "$decode_cmd" \
        exp/tri1/graph data/dev exp/tri1/decode &> exp/tri1/decode.log
) &
tri1decode=$!; # Grab the PID of the subshell
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri1/mkgraph.log and exp/tri1/decode.log for decoding outcomes"

echo -------------------------------------------------------------------------------
echo "Starting (medium) triphone training in exp/tri2 on" `date`
echo -------------------------------------------------------------------------------
steps/align_si.sh \
    --boost-silence 1.5 --nj 24 --cmd "$train_cmd" \
    data/train_sub3 data/lang exp/tri1 exp/tri1_ali_sub3 || exit 1
steps/train_deltas.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesTri2 $numGaussTri2 data/train_sub3 data/lang exp/tri1_ali_sub3 exp/tri2 || exit 1

echo -----------------------------------------------------------------------------------------
echo "Spawning decoding with (medium) triphone models in exp/tri2 on" `date`
echo -----------------------------------------------------------------------------------------
(
    mkdir -p exp/tri2/graph
    utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph &> exp/tri2/mkgraph.log
    mkdir -p exp/tri2/decode
    steps/decode.sh --nj $nj_decode --cmd "$decode_cmd" \
        exp/tri2/graph data/dev exp/tri2/decode &> exp/tri2/decode.log
) &
tri2decode=$!; # Grab the PID of the subshell
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri2/mkgraph.log and exp/tri2/decode.log for decoding outcomes"

echo -----------------------------------------------------------------------------
echo "Starting (full) triphone training in exp/tri3 on" `date`
echo -----------------------------------------------------------------------------
steps/align_si.sh \
    --boost-silence 1.5 --nj $nj_train --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1
steps/train_deltas.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesTri3 $numGaussTri3 data/train data/lang exp/tri2_ali exp/tri3 || exit 1

echo ---------------------------------------------------------------------------------------
echo "Spawning decoding with full triphone models in exp/tri3 on" `date`
echo ---------------------------------------------------------------------------------------
(
    mkdir -p exp/tri3/graph
    utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph &> exp/tri3/mkgraph.log
    mkdir -p exp/tri3/decode
    steps/decode.sh --nj $nj_decode --cmd "$decode_cmd" \
        exp/tri3/graph data/dev exp/tri3/decode &> exp/tri3/decode.log
) &
tri3decode=$!; # Grab the PID of the subshell
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri3/mkgraph.log and exp/tri3/decode.log for decoding outcomes"

echo ---------------------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
echo ---------------------------------------------------------------------------------
steps/align_si.sh \
    --boost-silence 1.5 --nj $nj_train --cmd "$train_cmd" \
    data/train data/lang exp/tri3 exp/tri3_ali || exit 1
steps/train_lda_mllt.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri3_ali exp/tri4 || exit 1

echo ----------------------------------------------------------------------------------
echo "Spawning decoding with lda_mllt models in exp/tri4 on" `date`
echo ----------------------------------------------------------------------------------
(
    mkdir -p exp/tri4/graph
    utils/mkgraph.sh \
        data/lang exp/tri4 exp/tri4/graph &> exp/tri4/mkgraph.log
    mkdir -p exp/tri4/decode
    steps/decode.sh --nj $nj_decode --cmd "$decode_cmd" \
        exp/tri4/graph data/dev exp/tri4/decode &> exp/tri4/decode.log
) &
tri4decode=$!; # Grab the PID of the subshell
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri4/mkgraph.log and exp/tri4/decode.log for decoding outcomes"

echo ----------------------------------------------------------------------------
echo "Starting (SAT) triphone training in exp/tri5 on" `date`
echo ----------------------------------------------------------------------------

steps/align_si.sh \
    --boost-silence 1.5 --nj $nj_train --cmd "$train_cmd" \
    data/train data/lang exp/tri4 exp/tri4_ali || exit 1
steps/train_sat.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train data/lang exp/tri4_ali exp/tri5 || exit 1

echo ------------------------------------------------------------------
echo "Spawning decoding with SAT models  on" `date`
echo ------------------------------------------------------------------
(
    mkdir -p exp/tri5/graph
    utils/mkgraph.sh \
        data/lang exp/tri5 exp/tri5/graph &> exp/tri5/mkgraph.log
    mkdir -p exp/tri5/decode
    touch exp/tri5/decode.started # A signal to the SGMM2 decoding step
    steps/decode_fmllr.sh --nj $nj_decode --cmd "$decode_cmd" \
        exp/tri5/graph data/dev exp/tri5/decode &> exp/tri5/decode.log \
    && touch exp/tri5/decode.finished # so SGMM2 decoding may proceed
) &
tri5decode=$!; # Grab the PID of the subshell; needed for SGMM2 decoding
sleep 30; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri5/mkgraph.log and exp/tri5/decode.log for decoding outcomes"

################################################################################
# Ready to start SGMM training
################################################################################

echo -------------------------------------------------
echo "Starting exp/ubm5 on" `date`
echo -------------------------------------------------
steps/align_fmllr.sh \
    --boost-silence 1.5 --nj $nj_train --cmd "$train_cmd" \
    data/train data/lang exp/tri5 exp/tri5_ali || exit 1
steps/train_ubm.sh \
    --cmd "$train_cmd" \
    $numGaussUBM data/train data/lang exp/tri5_ali exp/ubm5 || exit 1

echo --------------------------------------------------
echo "Starting exp/sgmm5 on" `date`
echo --------------------------------------------------
steps/train_sgmm2.sh \
    --cmd "$train_cmd" \
    $numLeavesSGMM $numGaussSGMM data/train data/lang exp/tri5_ali exp/ubm5/final.ubm exp/sgmm5 || exit 1

################################################################################
# Ready to decode with SGMM2 models
################################################################################

echo -----------------------------------------------------------------
echo "Spawning exp/sgmm5/decode[_fmllr] on" `date`
echo -----------------------------------------------------------------
(
    sleep 5; # Let the status message after the subshell get logged
    while [ ! -f exp/tri5/decode.started -o ! -f exp/tri5/decode.finished ]; do
        echo "exp/sgmm5/decode is waiting on SAT decoding ..." `date`
        sleep 5
    done
    while [ exp/tri5/decode.finished -ot exp/tri5/decode.started ]; do
        echo "exp/tri5/decode.finished is older than exp/tri5/decode.started"; \
        ls -lt exp/tri5/decode.finished exp/tri5/decode.started; \
        echo "Perhaps SAT decoding was restarted and is still running?"; \
        echo "exp/sgmm5/decode is still waiting on SAT decoding ..." `date`
        sleep 5
    done
    rm exp/tri5/decode.started exp/tri5/decode.finished
    mkdir -p exp/sgmm5/graph
    utils/mkgraph.sh \
        data/lang exp/sgmm5 exp/sgmm5/graph &> exp/sgmm5/mkgraph.log
    mkdir -p exp/sgmm5/decode
    steps/decode_sgmm2.sh \
        --nj $nj_decode --cmd "$decode_cmd" --transform-dir exp/tri5/decode \
        exp/sgmm5/graph data/dev/ exp/sgmm5/decode &> exp/sgmm5/decode.log
    steps/decode_sgmm2.sh --use-fmllr true --nj $nj_decode --cmd "$decode_cmd" \
        --transform-dir exp/tri5/decode \
        exp/sgmm5/graph data/dev/ exp/sgmm5/decode_fmllr &> exp/sgmm5/decode_fmllr.log
) &
sgmm5decode=$!; # Grab the PID of the subshell; needed for MMI rescoring
sleep 30; # Let any "start-up error" messages from the subshell get logged
echo "See exp/sgmm5/mkgraph.log, exp/sgmm5/decode.log and exp/sgmm5/decode_fmllr.log for decoding outcomes"

################################################################################
# Ready to start discriminative SGMM training
################################################################################

echo ------------------------------------------------------
echo "Starting exp/sgmm5_ali on" `date`
echo ------------------------------------------------------
steps/align_sgmm2.sh \
    --nj $nj_train --cmd "$train_cmd" --transform-dir exp/tri5_ali --use-graphs true --use-gselect true \
    data/train data/lang exp/sgmm5 exp/sgmm5_ali || exit 1

echo ----------------------------------------------------------
echo "Starting exp/sgmm5_denlats on" `date`
echo ----------------------------------------------------------
steps/make_denlats_sgmm2.sh \
    --nj $nj_train --sub-split $nj_train \
    --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" --transform-dir exp/tri5_ali \
    data/train data/lang exp/sgmm5_ali exp/sgmm5_denlats || exit 1

echo -----------------------------------------------------------
echo "Starting exp/sgmm5_mmi_b0.1 on" `date`
echo -----------------------------------------------------------
steps/train_mmi_sgmm2.sh \
    --cmd "$decode_cmd" --transform-dir exp/tri5_ali --boost 0.1 \
    data/train data/lang exp/sgmm5_ali exp/sgmm5_denlats \
    exp/sgmm5_mmi_b0.1 || exit 1

################################################################################
# Ready to decode with discriminative SGMM2 models
################################################################################

echo "exp/sgmm5_mmi_b0.1/decode will wait on PID $sgmm5decode if necessary"
wait $sgmm5decode; # Need lattices from the corresponding SGMM decoding passes
echo --------------------------------------------------------------------------
echo "Starting exp/sgmm5_mmi_b0.1/decode[_fmllr] on" `date`
echo --------------------------------------------------------------------------
for iter in 1 2 3 4; do
    steps/decode_sgmm2_rescore.sh \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode \
        data/lang data/dev exp/sgmm5/decode exp/sgmm5_mmi_b0.1/decode_it$iter
    steps/decode_sgmm2_rescore.sh \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode \
        data/lang data/dev exp/sgmm5/decode_fmllr exp/sgmm5_mmi_b0.1/decode_fmllr_it$iter
done

wait $tri1decode
wait $tri2decode
wait $tri3decode
wait $tri4decode

# No need to wait on $tri5decode ---> $sgmm5decode ---> sgmm5_mmi_b0.1decode

echo -----------------------------------------------------
echo "Finished successfully on" `date`
echo -----------------------------------------------------

exit 0
