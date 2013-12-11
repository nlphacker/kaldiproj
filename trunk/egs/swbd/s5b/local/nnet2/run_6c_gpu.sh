#!/bin/bash


# This script demonstrates discriminative training of neural nets.  It's on top
# of run_5c_gpu.sh, which uses adapted 40-dimensional features.  This version of
# the script uses GPUs.  We distinguish it by putting "_gpu" at the end of the
# directory name.


gpu_opts="-l gpu=1,hostname=g*"  # This is suitable for the CLSP network,
                                      # you'll likely have to change it.  we'll
                                      # use it later on, in the training (it's
                                      # not used in denlat creation)
stage=0
train_stage=-100

set -e # exit on error.

. cmd.sh
. utils/parse_options.sh


# The denominator lattice creation currently doesn't use GPUs.

# Note: we specify 1G each for the mem_free and ram_free which, is per
# thread... it will likely be less than the default.  Increase the beam relative
# to the defaults; this is just for this RM setup, where the default beams will
# likely generate very thin lattices.  Note: the transform-dir is important to
# specify, since this system is on top of fMLLR features.



if [ $stage -le 2 ]; then
  steps/nnet2/train_discriminative.sh --cmd "$decode_cmd"  --learning-rate 0.000002 \
    --modify-learning-rates true --last-layer-factor 0.1 \
    --num-epochs 4 --cleanup false \
    --num-jobs-nnet 4 --stage $train_stage \
    --transform-dir exp/tri4b \
    --num-threads 1 --parallel-opts "$gpu_opts" data/train data/lang \
    exp/nnet5c_gpu_ali exp/nnet5c_gpu_denlats exp/nnet5c_gpu/final.mdl exp/nnet6c_mpe_gpu
fi

if [ $stage -le 3 ]; then
  for epoch in 1 2 3 4; do 
    for lm_suffix in tg fsh_tgpr; do
      steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 --iter epoch$epoch \
        --config conf/decode.config --transform-dir exp/tri4b/decode_eval2000_sw1_${lm_suffix} \
        exp/tri4b/graph_sw1_${lm_suffix} data/eval2000 exp/nnet6c_mpe_gpu/decode_eval2000_sw1_${lm_suffix}_epoch$epoch &
    done
  done
fi



exit 0;
