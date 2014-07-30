export KALDI_ROOT=`pwd`/../../..
[ -f ${KALDI_ROOT}/tools/env.sh ] &&  . ${KALDI_ROOT}/tools/env.sh
#export KALDI_ROOT=/home/dpovey/kaldi-trunk-test
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk-2.4.8/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH
export LC_ALL=C
