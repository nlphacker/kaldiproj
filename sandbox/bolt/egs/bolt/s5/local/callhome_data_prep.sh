#!/bin/bash
# Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


. path.sh

if [ $# != 3 ]; then
   echo "Usage: callhome_data_prep.sh /path/to/speech_audio /path/to/transcription /path/to/destination"
   exit 1;
fi

AUDIO_DIR=$1
TRANS_DIR=$2
DESTINATION=$3

train_dir=$DESTINATION/train.callhome
devtest_dir=$DESTINATION/devtest.callhome
evltest_dir=$DESTINATION/evltest.callhome


mkdir -p $train_dir
mkdir -p $devtest_dir
mkdir -p $evltest_dir

#data directory check
if [ ! -d $AUDIO_DIR ] || [ ! -d $TRANS_DIR ]; then
  echo "Usage: callhome_data_prep.sh /path/to/speech_audio /path/to/transcription /path/to/destination"
  exit 1;
fi

#find sph audio file for train dev resp.
find $AUDIO_DIR -iname "*.sph" -ipath "*/train/*" > $train_dir/sph.flist
find $AUDIO_DIR -iname "*.sph" -ipath "*/devtest/*" > $devtest_dir/sph.flist
find $AUDIO_DIR -iname "*.sph" -ipath "*/evltest/*" > $evltest_dir/sph.flist

n=`cat $train_dir/sph.flist $devtest_dir/sph.flist | wc -l`
[ $n -ne 100 ] && \
  echo Warning: expected 100 data files, found $n


#We prepare the full kaldi directory (with the exception of the text file)
#directly in one pass through the data

for dataset in train devtest evltest ; do
  eval dest_dir=\$${dataset}_dir;
  echo -e "\n----Converting the CALLHOME $dataset dataset into kaldi directory in $dest_dir"
  find $TRANS_DIR -iname "*.txt" -ipath "*/$dataset/*" |\
    perl local/callhome_data_convert.pl $dest_dir/sph.flist $dest_dir || exit 1

  cat $dest_dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dest_dir/spk2utt || exit 1
done
echo "All CALLHOME datasets converted..."

#TODO: This should be moved to kaldi-tools directory
#transcripts normalization and segmentation
#(this needs external tools),
#Download and configure segment tools
pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:`pwd`/tools/mmseg-1.3.0/lib/python${pyver}/site-packages
if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
  echo "--- Downloading mmseg-1.3.0 ..."
  echo "NOTE: it assumes that you have Python, Setuptools installed on your system!"
  wget -P tools http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz
  tar xf tools/mmseg-1.3.0.tar.gz -C tools
  cd tools/mmseg-1.3.0
  mkdir -p lib/python${pyver}/site-packages
    python setup.py build
  python setup.py install --prefix=.
  cd ../..
  if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
    echo "mmseg is not found - installation failed?"
    exit 1
  fi
fi

#TODO: The text filtering should be improved
echo -e "\n----Preparing audio training transcripts in $train_dir"
cat $train_dir/transcripts.txt |\
  sed -e 's/\[[a-zA-Z]*_noise/[noise/g' |\
  sed -e 's/{[a-zA-Z]*_noise/{noise/g' |\
  sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
  sed '/^\s*$/d' |\
  local/callhome_normalize.pl |\
  python local/callhome_mmseg_segment.py |\
  awk '{if (NF > 1) print $0;}' | sort -u > $train_dir/text
utils/fix_data_dir.sh $train_dir

for dataset in $devtest_dir $evltest_dir; do
  echo -e "\n----Preparing audio (reference) transcripts in $dataset"
  cat $dataset/transcripts.txt |\
    sed -e 's/\[[a-zA-Z]*_noise/[noise/g' |\
    sed -e 's/{[a-zA-Z]*_noise/{noise/g' |\
    sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
    local/callhome_normalize.pl |\
    python local/callhome_mmseg_segment.py |\
    awk '{if (NF > 1) print $0;}' | sort -u > $dataset/text
  utils/fix_data_dir.sh $dataset
done

echo -e "\n\nCALLHOME data preparation succeeded..."
