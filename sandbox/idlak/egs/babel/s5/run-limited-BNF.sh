#!/bin/bash

# System and data directories
#SCRIPT=$(readlink -f $0)
#SysDir=`dirname $SCRIPT`
SysDir=`pwd`
echo $SysDir

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source train and decode cmds.
. parse_options.sh || exit 1;

configfile=$1
[ -f $configfile ] && . $configfile 
[ -f ./local.conf ] && . ./local.conf

################################################################################
# Preparation
################################################################################
# Install pfiles_utils and set the commands path in the 
# PATH environment variable or in the path.sh
# http://www.icsi.berkeley.edu/ftp/pub/real/davidj/pfile_utils-v0_51.tar.gz

# On the machines with GPUs, use ptdnn for DNN training.
# These machines should have Theano installed. Refer to README in ptdnn 
# for installation information.

################################################################################
# Ready to start DNN training
################################################################################

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/bnf_dnn on" `date`
echo ---------------------------------------------------------------------
# Note that align_fmllr.sh may have been implemented in run-limited.sh
steps/align_fmllr.sh --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri4 exp/tri4_ali || exit 1
steps_BNF/build_nnet_pfile.sh --cmd "$train_cmd" \
    data/train data/lang exp/tri4_ali exp_BNF/bnf_dnn || exit 1
# Now you can copy train.pfile.gz and valid.pfile.gz to the GPU machine and run
# ptdnn

################################################################################
# Ready to make BNF features
################################################################################

# Assume that final.nnet.kaldi generated by ptdnn is copied back to exp_BNF/bnf_dnn
# and renamed into final.nnet
echo ---------------------------------------------------------------------
echo "Starting making BNF in exp_BNF/make_bnf on" `date`
echo ---------------------------------------------------------------------
steps_BNF/make_bnf_feat.sh --nj $train_nj --cmd "$train_cmd" --transform_dir exp/tri4_ali \
    data/train data/train_bnf exp_BNF/bnf_dnn exp/tri4_ali exp_BNF/make_bnf || exit 1
steps_BNF/make_bnf_feat.sh --nj $decode_nj --cmd "$train_cmd" --transform_dir exp/tri4/decode \
    data/dev data/dev_bnf exp_BNF/bnf_dnn exp/tri4_ali exp_BNF/make_bnf || exit 1

################################################################################
# Complete data/train_bnf data/dev_bnf 
################################################################################
# Copy everything [except feats.scp] from data/train to data/train_bnf, and from
# data/dev to data/dev_bnf. Once I know clearly what are in your data/train and
# data/dev, I can automate this step in the script. 

################################################################################
# Ready to start triphone training on BNF 
################################################################################

echo ---------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp_BNF/tri5 on" `date`
echo ---------------------------------------------------------------------
# Here we start from fmllr_align in exp/tri4_ali
steps_BNF/train_lda_mllt.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train_bnf data/lang exp/tri4_ali exp_BNF/tri5 || exit 1

echo ---------------------------------------------------------------------
echo "Spawning decoding with lda_mllt models in exp_BNF/tri5 on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp_BNF/tri5/graph
    utils/mkgraph.sh \
        data/lang exp_BNF/tri5 exp_BNF/tri5/graph &> exp_BNF/tri5/mkgraph.log
    mkdir -p exp_BNF/tri5/decode
    steps_BNF/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp_BNF/tri5/graph data/dev_bnf exp_BNF/tri5/decode &> exp_BNF/tri5/decode.log
    
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev_bnf exp_BNF/tri5/decode
) &
echo "See exp_BNF/tri5/mkgraph.log and exp_BNF/tri5/decode.log for decoding outcomes"

################################################################################
# Ready to start SGMM training
################################################################################

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/ubm6 on" `date`
echo ---------------------------------------------------------------------
steps_BNF/align_si.sh \
    --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
    data/train_bnf data/lang exp_BNF/tri5 exp_BNF/tri5_ali || exit 1
steps_BNF/train_ubm.sh --cmd "$train_cmd" \
    $numGaussUBM data/train_bnf data/lang exp_BNF/tri5_ali exp_BNF/ubm6 || exit 1

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/sgmm6 on" `date`
echo ---------------------------------------------------------------------
steps_BNF/train_sgmm2.sh --cmd "$train_cmd" \
    $numLeavesSGMM $numGaussSGMM data/train_bnf data/lang exp_BNF/tri5_ali exp_BNF/ubm6/final.ubm exp_BNF/sgmm6 || exit 1

################################################################################
# Ready to decode with SGMM2 models
################################################################################

echo ---------------------------------------------------------------------
echo "Spawning exp_BNF/sgmm6/decode[_fmllr] on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp_BNF/sgmm6/graph
    utils/mkgraph.sh \
        data/lang exp_BNF/sgmm6 exp_BNF/sgmm6/graph &> exp_BNF/sgmm6/mkgraph.log
    mkdir -p exp_BNF/sgmm6/decode
    steps_BNF/decode_sgmm2.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp_BNF/sgmm6/graph data/dev_bnf exp_BNF/sgmm6/decode &> exp_BNF/sgmm6/decode.log
    steps_BNF/decode_sgmm2.sh --use-fmllr true --nj $decode_nj --cmd "$decode_cmd" \
        exp_BNF/sgmm6/graph data/dev_bnf exp_BNF/sgmm6/decode_fmllr &> exp_BNF/sgmm6/decode_fmllr.log
    
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev_bnf exp_BNF/sgmm6/decode
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev_bnf exp_BNF/sgmm6/decode_fmllr
) &
sgmm6decode=$!; # Grab the PID of the subshell; needed for MMI rescoring
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp_BNF/sgmm6/mkgraph.log, exp_BNF/sgmm6/decode.log and exp_BNF/sgmm6/decode_fmllr.log for decoding outcomes"

################################################################################
# Ready to start discriminative SGMM training
################################################################################

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/sgmm6_ali on" `date`
echo ---------------------------------------------------------------------
steps_BNF/align_sgmm2.sh \
    --nj $train_nj --cmd "$train_cmd" --use-graphs true --use-gselect true \
    data/train_bnf data/lang exp_BNF/sgmm6 exp_BNF/sgmm6_ali || exit 1

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/sgmm6_denlats on" `date`
echo ---------------------------------------------------------------------
# more experiments need to be done on the lat-beam
steps_BNF/make_denlats_sgmm2.sh \
    --nj $train_nj --sub-split $train_nj \
    --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" 
    data/train_bnf data/lang exp_BNF/sgmm6_ali exp_BNF/sgmm6_denlats || exit 1

echo ---------------------------------------------------------------------
echo "Starting exp_BNF/sgmm6_mmi_b0.1 on" `date`
echo ---------------------------------------------------------------------
steps_BNF/train_mmi_sgmm2.sh \
    --cmd "$decode_cmd" --boost 0.1 \
    data/train_bnf data/lang exp_BNF/sgmm6_ali exp_BNF/sgmm6_denlats \
    exp_BNF/sgmm6_mmi_b0.1 || exit 1

################################################################################
# Ready to decode with discriminative SGMM2 models
################################################################################

echo "exp_BNF/sgmm6_mmi_b0.1/decode will wait on PID $sgmm6decode if necessary"
wait $sgmm6decode; # Need lattices from the corresponding SGMM decoding passes
echo ---------------------------------------------------------------------
echo "Starting exp_BNF/sgmm6_mmi_b0.1/decode[_fmllr] on" `date`
echo ---------------------------------------------------------------------
for iter in 1 2 3 4; do
    steps_BNF/decode_sgmm2_rescore.sh \
        --cmd "$decode_cmd" --iter $iter \
        data/lang data/dev_bnf exp_BNF/sgmm6/decode exp_BNF/sgmm6_mmi_b0.1/decode_it$iter
    steps_BNF/decode_sgmm2_rescore.sh \
        --cmd "$decode_cmd" --iter $iter \
        data/lang data/dev_bnf exp_BNF/sgmm6/decode_fmllr exp_BNF/sgmm6_mmi_b0.1/decode_fmllr_it$iter
    
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev_bnf exp_BNF/sgmm6_mmi_b0.1/decode_it$iter
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev_bnf exp_BNF/sgmm6_mmi_b0.1/decode_fmllr_it$iter
done


wait 

# No need to wait on $tri4decode ---> $sgmm5decode ---> sgmm5_mmi_b0.1decode

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
