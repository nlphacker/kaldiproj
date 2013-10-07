#!/bin/bash

# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
# Apache 2.0.

if [ $# -ne 1 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

[ -f $conf/test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf/dev_spk.list ] || error_exit "$PROG: dev-set speaker list not found.";

if [ ! -d $*/TRAIN -o ! -d $*/TEST ]; then
  echo "timit_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to TIMIT directory"
  echo "with name like /export/corpora5/LDC/LDC93S1/timit/TIMIT"
  exit 1;
fi

tmpdir=$(mktemp -d);
trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers. The list of speakers in the 24-speaker core test 
# set and the 50-speaker development set must be supplied to the script. All
# speakers in the 'train' directory are used for training.
tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk    # Just in case!
tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk  # Just in case!

ls -d "$*"/TRAIN/DR*/* | sed -e "s:^.*/::" > $tmpdir/train_spk

cd $dir
for x in train dev test; do
  # First, find the list of audio files (use only si & sx utterances).
  # Note: train & test sets are under different directories, but doing find on 
  # both and grepping for the speakers will work correctly.

  find $*/{TRAIN,TEST} -not \( -name 'SA*' \) -name '*.WAV' \
    | grep -f $tmpdir/${x}_spk > ${x}_sph.flist

  sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:' ${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
  paste $tmpdir/${x}_sph.uttids ${x}_sph.flist \
    | sort -k1,1 > ${x}_sph.scp

  cat ${x}_sph.scp | awk '{print $1}' > ${x}.uttids

  # Now, Convert the transcripts into our format (no normalization yet)
  # Get the transcripts: each line of the output contains an utterance 
  # ID followed by the transcript.
  find $*/{TRAIN,TEST} -not \( -name 'SA*' \) -name '*.PHN' \
    | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_phn.flist
  sed -e 's:.*/\(.*\)/\(.*\).PHN$:\1_\2:' $tmpdir/${x}_phn.flist \
    > $tmpdir/${x}_phn.uttids
  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    cut -f3 -d' ' "$line" | tr '\n' ' ' | sed -e 's: *$:\n:'
  done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans
  paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans \
    | sort -k1,1 > ${x}.trans

# Do normalization steps. 
  cat ${x}.trans | $local/timit_norm_trans.pl -i - -m $conf/phones.60-48-39.map -to 39 | sort > $x.txt || exit 1;

  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp

# Make the utt2spk and spk2utt files.

  cut -f1 -d'_'  $x.uttids | paste -d' ' $x.uttids - > $x.utt2spk 
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;

  cat $x.spk2utt | awk '{print $1}' | perl -ane 'chop; m:^.:; print "$_ $&\n";' > $x.spk2gender
done

echo "Data preparation succeeded"