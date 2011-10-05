#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


if [ $# != 2 ]; then
  echo "Usage: split_data.sh data-dir num-to-split"
  exit 1
fi

data=$1
numsplit=$2

if [ $numsplit -le 0 ]; then
  echo "Invalid num-split argument $numsplit";
  exit 1;
fi

n=0;
feats=""
wavs=""
utt2spks=""
texts=""

# `scripts/get_split.pl` returns "0 1 2 3" or "00 01 .. 18 19" or whatever.
for n in `scripts/get_splits.pl $numsplit`; do
   mkdir -p $data/split$numsplit/$n
   feats="$feats $data/split$numsplit/$n/feats.scp"
   wavs="$wavs $data/split$numsplit/$n/wavs.scp"
   texts="$texts $data/split$numsplit/$n/text"
   utt2spks="$utt2spks $data/split$numsplit/$n/utt2spk"
done

scripts/split_scp.pl --utt2spk=$data/utt2spk $data/utt2spk $utt2spks
scripts/split_scp.pl --utt2spk=$data/utt2spk $data/feats.scp $feats
scripts/split_scp.pl --utt2spk=$data/utt2spk $data/wav.scp $wavs
scripts/split_scp.pl --utt2spk=$data/utt2spk $data/text $texts

for n in `scripts/get_splits.pl $numsplit`; do
   scripts/utt2spk_to_spk2utt.pl $data/split$numsplit/$n/utt2spk > $data/split$numsplit/$n/spk2utt
   # for completeness, also split the spk2gender file
   scripts/filter_scp.pl $data/split$numsplit/$n/spk2utt $data/spk2gender > $data/split$numsplit/$n/spk2gender
done
