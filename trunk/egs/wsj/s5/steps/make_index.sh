#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0

# Begin configuration section.  
model= # You can specify the model to use
cmd=run.pl
acwt=0.083333
lmwt=1.0
max_silence_frames=50
strict=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/make_index.sh [options] <kws-data-dir> <lang-dir> <decode-dir> <kws-dir>"
   echo "... where <decode-dir> is where you have the lattices, and is assumed to be"
   echo " a sub-directory of the directory where the model is."
   echo "e.g.: steps/make_index.sh data/kws data/lang exp/sgmm2_5a_mmi/decode/ exp/sgmm2_5a_mmi/decode/kws/"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --acwt <float>                                   # acoustic scale used for lattice"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --lmwt <float>                                   # lm scale used for lattice"
   echo "  --model <model>                                  # which model to use"
   echo "                                                   # speaker-adapted decoding"
   echo "  --max-silence-frames <int>                       # maximum #frames for silence"
   exit 1;
fi


kwsdatadir=$1;
langdir=$2;
decodedir=$3;
kwsdir=$4;
srcdir=`dirname $decodedir`; # The model directory is one level up from decoding directory.

mkdir -p $kwsdir/log;
nj=`cat $decodedir/num_jobs` || exit 1;
echo $nj > $kwsdir/num_jobs;
word_boundary=$langdir/phones/word_boundary.int
utter_id=$kwsdatadir/utter_id

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  model=$srcdir/final.mdl; 
fi

for f in $word_boundary $model $decodedir/lat.1.gz; do
  [ ! -f $f ] && echo "make_index.sh: no such file $f" && exit 1;
done

$cmd JOB=1:$nj $kwsdir/log/index.JOB.log \
 lattice-align-words $word_boundary $model "ark:gzip -cdf $decodedir/lat.JOB.gz|" ark:- \| \
   lattice-scale --acoustic-scale=$acwt --lm-scale=$lmwt ark:- ark:- \| \
   lattice-to-kws-index --strict=$strict ark:$utter_id ark:- ark:- \| \
   kws-index-union --strict=$strict ark:- "ark:|gzip -c > $kwsdir/index.JOB.gz"

exit 0;
