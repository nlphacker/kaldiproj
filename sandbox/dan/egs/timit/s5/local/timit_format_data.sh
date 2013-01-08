#!/bin/bash

# Copyright 2012  Navdeep Jaitly
# Copyright 2010-2011  Microsoft Corporation

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

# To be run from one directory above this script.
# I have cleaned up the version from s3 to account for the fact that
# this recipe is a simple monophone system with a biphone language model.
# Hence I have taken out the parts for disambiguation symbols. Also,
# I have simplified the lexicon fst. It not emits and espilon when it 
# sees a silence phone, and there is no cost associated with this.


if [ -f path.sh ]; then . path.sh; fi

arpa_lm=data/local/lm/biphone/lm_unpruned.gz

data_list="train test dev"

for x in lang $data_list; do
  mkdir -p data/$x
done

# Copy stuff into its final location:

for x in $data_list; do
  cp data/local/$x.spk2utt data/$x/spk2utt || exit 1;
  cp data/local/$x.utt2spk data/$x/utt2spk || exit 1;
  cp data/local/${x}_wav.scp data/$x/wav.scp || exit 1;
  cp data/local/${x}_trans.txt data/$x/text || exit 1;
  scripts/filter_scp.pl data/$x/spk2utt data/local/spk2gender.map > data/$x/spk2gender || exit 1;
done


scripts/make_words_symtab.pl < data/local/lexicon.txt > data/lang/words.txt
scripts/make_phones_symtab.pl < data/local/lexicon.txt > data/lang/phones.txt

silphones="sil";
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/lang/phones.txt "$silphones" data/lang/silphones.csl \
  data/lang/nonsilphones.csl

echo "Creating L.fst"
#scripts/make_phone_lexicon_fst.pl data/local/lexicon.txt sil | \
scripts/make_lexicon_fst.pl data/local/lexicon.txt sil 0.5| \
  fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/lang/L.fst
echo "Done creating L.fst"

echo "Creating G.fst"
gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   scripts/s2eps.pl | \
   fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt  --keep_isymbols=false \
        --keep_osymbols=false > data/lang/G.fst

echo "G.fst created. How stochastic is it ?"
fstisstochastic data/lang/G.fst 

# Checking that G.fst is determinizable.
fstdeterminize data/lang/G.fst /dev/null || echo Error determinizing G.

# Checking that LG is stochastic:
echo "How stochastic is LG.fst."
fsttablecompose data/lang/L.fst data/lang/G.fst | \
   fstisstochastic 

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head


silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > data/lang/topo 

echo timit_format_data succeeded.
