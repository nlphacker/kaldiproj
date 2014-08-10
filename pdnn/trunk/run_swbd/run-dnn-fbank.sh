#!/bin/bash

# Apache 2.0
# This is the script that trains DNN system over the filterbank features. It
# is to  be  run after run.sh. Before running this, you should already build
# the initial GMM model. This script requires a GPU card, and also the "pdnn"
# toolkit to train the DNN. The input filterbank features are with mean  and
# variance normalization. 

# For more informaiton regarding the recipes and results, visit our webiste
# http://www.cs.cmu.edu/~ymiao/kaldipdnn

working_dir=exp_pdnn/dnn_fbank
do_ptr=true    # whether to do pre-training
delete_pfile=true # whether to delete pfiles after DNN training

gmmdir=exp/tri4b

# Specify the gpu device to be used
gpu=gpu

cmd=run.pl
. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

# At this point you may want to make sure the directory $working_dir is
# somewhere with a lot of space, preferably on the local GPU-containing machine.
if [ ! -d pdnn ]; then
  echo "Checking out PDNN code."
  svn co svn://svn.code.sf.net/p/kaldipdnn/code-0/pdnn pdnn
fi

if [ ! -d steps_pdnn ]; then
  echo "Checking out steps_pdnn scripts."
  svn co svn://svn.code.sf.net/p/kaldipdnn/code-0/trunk/steps_pdnn steps_pdnn
fi

if ! nvidia-smi; then
  echo "The command nvidia-smi was not found: this probably means you don't have a GPU."
  echo "(Note: this script might still work, it would just be slower.)"
fi

# The hope here is that Theano has been installed either to python or to python2.6
pythonCMD=python
if ! python -c 'import theano;'; then
  if ! python2.6 -c 'import theano;'; then
    echo "Theano does not seem to be installed on your machine.  Not continuing."
    echo "(Note: this script might still work, it would just be slower.)"
    exit 1;
  else
    pythonCMD=python2.6
  fi
fi

mkdir -p $working_dir/log

! gmm-info $gmmdir/final.mdl >&/dev/null && \
   echo "Error getting GMM info from $gmmdir/final.mdl" && exit 1;

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;

echo ---------------------------------------------------------------------
echo "Creating DNN training and validation data (pfiles)"
echo ---------------------------------------------------------------------
# Alignment on the training and validation data
if [ ! -d ${gmmdir}_ali_nodup ]; then
  echo "Generate alignment on train data"
  steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
    data/train_nodup data/lang $gmmdir ${gmmdir}_ali_nodup || exit 1
fi
if [ ! -d ${gmmdir}_ali_dev ]; then
  echo "Generate alignment on valid data"
  steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
    data/train_dev data/lang $gmmdir ${gmmdir}_ali_dev || exit 1
fi

# Generate the fbank features. We generate the 40-dimensional fbanks on each frame
echo "--num-mel-bins=40" > conf/fbank.conf
echo "--sample-frequency=8000" >> conf/fbank.conf
mkdir -p $working_dir/data
if [ ! -d $working_dir/data/train ]; then
  echo "Save fbank features of train data"
  cp -r data/train_nodup $working_dir/data/train
  ( cd $working_dir/data/train; rm -rf {cmvn,feats}.scp split*; )
  steps/make_fbank.sh --cmd "$train_cmd" --nj 24 $working_dir/data/train $working_dir/_log $working_dir/_fbank || exit 1;
  utils/fix_data_dir.sh $working_dir/data/train || exit;
  steps/compute_cmvn_stats.sh $working_dir/data/train $working_dir/_log $working_dir/_fbank || exit 1;
fi
if [ ! -d $working_dir/data/valid ]; then
  echo "Save fbank features of valid data"
  cp -r data/train_dev $working_dir/data/valid
  ( cd $working_dir/data/valid; rm -rf {cmvn,feats}.scp split*; )
  steps/make_fbank.sh --cmd "$train_cmd" --nj 24 $working_dir/data/valid $working_dir/_log $working_dir/_fbank || exit 1;
  utils/fix_data_dir.sh $working_dir/data/valid || exit;
  steps/compute_cmvn_stats.sh $working_dir/data/valid $working_dir/_log $working_dir/_fbank || exit 1;
fi
if [ ! -d $working_dir/data/eval2000 ]; then
  echo "Save fbank features of eval2000"
  cp -r data/eval2000 $working_dir/data/eval2000
  ( cd $working_dir/data/eval2000; rm -rf {cmvn,feats}.scp split*; )
  steps/make_fbank.sh --cmd "$train_cmd" --nj 24 $working_dir/data/eval2000 $working_dir/_log $working_dir/_fbank || exit 1;
  utils/fix_data_dir.sh $working_dir/data/eval2000 || exit;
  steps/compute_cmvn_stats.sh $working_dir/data/eval2000 $working_dir/_log $working_dir/_fbank || exit 1;
fi

# By default, inputs include 11 frames (+/-5) of 40-dimensional log-scale filter-banks, with 440 dimensions.
if [ ! -f $working_dir/train.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --every-nth-frame 1 --do-split false \
    --norm-vars true --splice-opts "--left-context=5 --right-context=5" --input-dim 440 \
    $working_dir/data/train ${gmmdir}_ali_nodup $working_dir || exit 1
  ( cd $working_dir; mv concat.pfile train.pfile; )
  touch $working_dir/train.pfile.done
fi
if [ ! -f $working_dir/valid.pfile.done ]; then
  steps_pdnn/build_nnet_pfile.sh --cmd "$train_cmd" --every-nth-frame 1 --do-split false \
    --norm-vars true --splice-opts "--left-context=5 --right-context=5" --input-dim 440 \
    $working_dir/data/valid ${gmmdir}_ali_dev $working_dir || exit 1
  ( cd $working_dir; mv concat.pfile valid.pfile; )
  touch $working_dir/valid.pfile.done
fi

echo ---------------------------------------------------------------------
echo "Starting DNN training"
echo ---------------------------------------------------------------------
feat_dim=$(cat $working_dir/train.pfile |head |grep num_features| awk '{print $2}') || exit 1;

if $do_ptr && [ ! -f $working_dir/dnn.ptr.done ]; then
  echo "SDA Pre-training"
  $cmd $working_dir/log/dnn.ptr.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/run_SdA.py --train-data "$working_dir/train.pfile,partition=2000m,random=true,stream=true" \
                          --nnet-spec "$feat_dim:2048:2048:2048:2048:2048:2048:2048:$num_pdfs" \
                          --first-reconstruct-activation "tanh" \
                          --wdir $working_dir --output-file $working_dir/dnn.ptr \
                          --ptr-layer-number 7 --epoch-number 5 || exit 1;
  touch $working_dir/dnn.ptr.done
fi

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/ptdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    $pythonCMD pdnn/run_DNN.py --train-data "$working_dir/train.pfile,partition=2000m,random=true,stream=true" \
                          --valid-data "$working_dir/valid.pfile,partition=600m,random=true,stream=true" \
                          --nnet-spec "$feat_dim:2048:2048:2048:2048:2048:2048:2048:$num_pdfs" \
                          --ptr-file $working_dir/dnn.ptr --ptr-layer-number 7 \
                          --output-format kaldi --lrate "D:0.08:0.5:0.2,0.2:8" \
                          --wdir $working_dir --output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
  $delete_pfile && rm -rf $working_dir/*.pfile
fi

echo ---------------------------------------------------------------------
echo "Decode the final system"
echo ---------------------------------------------------------------------
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph_sw1_tg
  steps_pdnn/decode_dnn.sh --nj 24 --scoring-opts "--min-lmwt 7 --max-lmwt 18" --cmd "$decode_cmd" --norm-vars true \
     $graph_dir $working_dir/data/eval2000 ${gmmdir}_ali_nodup $working_dir/decode_eval2000_sw1_tg || exit 1;
  touch $working_dir/decode.done
fi

echo "Finish !!"
