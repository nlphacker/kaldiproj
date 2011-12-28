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

# Call this script from one level above, e.g. from the s3/ directory.  It puts
# its output in data/local/.

# This script takes as arguments the locations of the WSJ disks from 
# LDC, and creates various files in data/local/, e.g. lists of files,
# and transcripts.  This script also does some transcript normalization,
# and maps back and forth between speaker-ids and utterance-ids.
# Most of these files are not used directly by Kaldi tools; the 
# script wsj_format_data.sh creates things that are directly
# used by Kaldi.
# The following illustrates some of the file formats:
# head -1 train_si284*    
# ==> train_si284.flist <==
# /mnt/matylda2/data/WSJ0/11-1.1/wsj0/si_tr_s/011/011c0201.wv1

# ==> train_si284.spk2utt <==
# 011 011c0201 011c0202 011c0203 011c0204 011c0205 011c0206 011c0207 011c0208 011c0209 011c020a 011c020b 011c020c 011c020d 011c020e 011c020f 011c020g 011c020h 011c020i 011c020j 011c020k 011c020l 011c020m 011c020n 011c020o 011c020p 011c020q 011c020r 011c020s 011c020t 011c020u 011c020v 011c020w 011c020x 011c020y 011c020z 011c0210 011c0211 011c0212 011c0213 011c0214 011c0215 011c0216 011c0217 011c0218 011c0219 011c021a 011c021b 011c021c 011c021d 011c021e 011o0301 011o0302 011o0303 011o0304 011o0305 011o0306 011o0307 011o0308 011o0309 011o030a 011o030b 011o030c 011o030d 011o030e 011o030f 011o030g 011o030h 011o030i 011o030j 011o030k 011o030l 011o030m 011o030n 011o030o 011o030p 011o030q 011o030r 011o030s 011o030t 011o030u 011o030v 011o030w 011o030x 011o030y 011o030z 011o0310 011o0311 011o0312 011o0313 011o0314 011o0315 011o0316 011o0317 011o0318 011o0319 011o031a 011o031b 011o031c 011o031d 011o031e 011o031f

# ==> train_si284.trans1 <==
# 011c0201 The sale of the hotels is part of Holiday\'s strategy to sell off assets and concentrate on property management 

# ==> train_si284.txt <==
# 011c0201 THE SALE OF THE HOTELS IS PART OF HOLIDAY'S STRATEGY TO SELL OFF ASSETS AND CONCENTRATE ON PROPERTY MANAGEMENT

# ==> train_si284.utt2spk <==
# 011c0201 011

# ==> train_si284_sph.scp <==
# 011c0201 /mnt/matylda2/data/WSJ0/11-1.1/wsj0/si_tr_s/011/011c0201.wv1

# ==> train_si284_wav.scp <==
# 011c0201 /homes/eva/q/qpovey/sourceforge/kaldi/trunk/tools/sph2pipe_v2.5/sph2pipe -f wav /mnt/matylda2/data/WSJ0/11-1.1/wsj0/si_tr_s/011/011c0201.wv1 |


if [ $# -lt 4 ]; then
   echo "Too few arguments to wsj_data_prep.sh: need a list of WSJ directories ending e.g. 11-13.1"
   exit 1;
fi

mkdir -p data/local
local=`pwd`/local
scripts=`pwd`/scripts

sph2pipe=`cd ../../..; echo $PWD/tools/sph2pipe_v2.5/sph2pipe`
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi
export PATH=$PATH:`pwd`/../../../tools/irstlm/bin

cd data/local


# Make directory of links to the WSJ disks such as 11-13.1.  This relies on the command
# line arguments being absolute pathnames.
rm -r links/ 2>/dev/null
mkdir links/
ln -s $* links

# Do some basic checks that we have what we expected.
if [ ! -d links/11-13.1 -o ! -d links/13-34.1 -o ! -d links/11-2.1 ]; then
  echo "wsj_data_prep.sh: Spot check of command line arguments failed"
  echo "Command line arguments must be absolute pathnames to WSJ directories"
  echo "with names like 11-13.1."
  exit 1;
fi

# This version for SI-84

cat links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 $local/ndx2flist.pl $* | sort | \
 grep -v 11-2.1/wsj0/si_tr_s/401 > train_si84.flist

# This version for SI-284
cat links/13-34.1/wsj1/doc/indices/si_tr_s.ndx \
 links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 $local/ndx2flist.pl  $* | sort | \
 grep -v 11-2.1/wsj0/si_tr_s/401 > train_si284.flist


# Now for the test sets.
# links/13-34.1/wsj1/doc/indices/readme.doc 
# describes all the different test sets.
# Note: each test-set seems to come in multiple versions depending
# on different vocabulary sizes, verbalized vs. non-verbalized
# pronunciations, etc.  We use the largest vocab and non-verbalized
# pronunciations.
# The most normal one seems to be the "baseline 60k test set", which
# is h1_p0. 

# Nov'92 (333 utts)
# These index files have a slightly different format;
# have to add .wv1
cat links/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
  $local/ndx2flist.pl $* |  awk '{printf("%s.wv1\n", $1)}' | \
  sort > test_eval92.flist

# Nov'92 (330 utts, 5k vocab)
cat links/11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx | \
  $local/ndx2flist.pl $* |  awk '{printf("%s.wv1\n", $1)}' | \
  sort > test_eval92_5k.flist

# Nov'93: (213 utts)
# Have to replace a wrong disk-id.
cat links/13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx | \
  sed s/13_32_1/13_33_1/ | \
  $local/ndx2flist.pl $* | sort > test_eval93.flist

# Nov'93: (213 utts, 5k)
cat links/13-32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx | \
  sed s/13_32_1/13_33_1/ | \
  $local/ndx2flist.pl $* | sort > test_eval93_5k.flist

# Dev-set for Nov'93 (503 utts)
cat links/13-34.1/wsj1/doc/indices/h1_p0.ndx | \
  $local/ndx2flist.pl $* | sort > test_dev93.flist

# Dev-set for Nov'93 (513 utts, 5k vocab)
cat links/13-34.1/wsj1/doc/indices/h2_p0.ndx | \
  $local/ndx2flist.pl $* | sort > test_dev93_5k.flist


# Dev-set Hub 1,2 (503, 913 utterances)

# Note: the ???'s below match WSJ and SI_DT, or wsj and si_dt.  
# Sometimes this gets copied from the CD's with upcasing, don't know 
# why (could be older versions of the disks).
find `readlink links/13-16.1`/???1/??_??_20 -print | grep ".WV1" | sort > dev_dt_20.flist
find `readlink links/13-16.1`/???1/??_??_05 -print | grep ".WV1" | sort > dev_dt_05.flist


# Finding the transcript files:
for x in $*; do find -L $x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
   $local/flist2scp.pl $x.flist | sort > ${x}_sph.scp
   cat ${x}_sph.scp | awk '{print $1}' | $local/find_transcripts.pl  dot_files.flist > $x.trans1
done

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
   cat $x.trans1 | $local/normalize_transcript.pl $noiseword | sort > $x.txt || exit 1;
done

 
# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp
done

# Make the utt2spk and spk2utt files.
for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
   cat ${x}_sph.scp | awk '{print $1}' | perl -ane 'chop; m:^...:; print "$_ $&\n";' > $x.utt2spk
   cat $x.utt2spk | $scripts/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done


#in case we want to limit lm's on most frequent words, copy lm training word frequency list
cp links/13-32.1/wsj1/doc/lng_modl/vocab/wfl_64.lst .

# The 20K vocab, open-vocabulary language model (i.e. the one with UNK), without
# verbalized pronunciations.   This is the most common test setup, I understand.

cp links/13-32.1/wsj1/doc/lng_modl/base_lm/bcb20onp.z lm_bg.arpa.gz || exit 1;
chmod u+w lm_bg.arpa.gz

# trigram would be:
cat links/13-32.1/wsj1/doc/lng_modl/base_lm/tcb20onp.z | \
 perl -e 'while(<>){ if(m/^\\data\\/){ print; last;  } } while(<>){ print; }' | \
 gzip -c -f > lm_tg.arpa.gz || exit 1;

prune-lm --threshold=1e-7 lm_tg.arpa.gz lm_tgpr.arpa || exit 1;
gzip -f lm_tgpr.arpa || exit 1;

# repeat for 5k language models
cp links/13-32.1/wsj1/doc/lng_modl/base_lm/bcb05onp.z  lm_bg_5k.arpa.gz || exit 1;
chmod u+w lm_bg_5k.arpa.gz

# trigram would be: !only closed vocabulary here!
cp links/13-32.1/wsj1/doc/lng_modl/base_lm/tcb05cnp.z lm_tg_5k.arpa.gz || exit 1;
chmod u+w lm_tg_5k.arpa.gz
gunzip lm_tg_5k.arpa.gz
tail -n 4328839 lm_tg_5k.arpa | gzip -c -f > lm_tg_5k.arpa.gz
rm lm_tg_5k.arpa

prune-lm --threshold=1e-7 lm_tg_5k.arpa.gz lm_tgpr_5k.arpa || exit 1;
gzip -f lm_tgpr_5k.arpa || exit 1;


if [ ! -f wsj0-train-spkrinfo.txt ]; then
  wget http://www.ldc.upenn.edu/Catalog/docs/LDC93S6A/wsj0-train-spkrinfo.txt
fi

if [ ! -f wsj0-train-spkrinfo.txt ]; then
  echo "Could not get the spkrinfo.txt file from LDC website (moved)?"
  echo "This is possibly omitted from the training disks; couldn't find it." 
  echo "Everything else may have worked; we just may be missing gender info"
  echo "which is only needed for VTLN-related diagnostics anyway."
  exit 1
fi
# Note: wsj0-train-spkrinfo.txt doesn't seem to be on the disks but the
# LDC put it on the web.  Perhaps it was accidentally omitted from the
# disks.  

cat links/11-13.1/wsj0/doc/spkrinfo.txt \
    links/13-32.1/wsj1/doc/evl_spok/spkrinfo.txt \
    links/13-34.1/wsj1/doc/dev_spok/spkrinfo.txt \
    links/13-34.1/wsj1/doc/train/spkrinfo.txt \
   ./wsj0-train-spkrinfo.txt  | \
    perl -ane 'tr/A-Z/a-z/; m/^;/ || print;' | \
   awk '{print $1, $2}' | grep -v -- -- | sort | uniq > spk2gender.map


echo "Data preparation succeeded"
