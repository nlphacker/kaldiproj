#!/bin/bash
#
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

# To be run from one directory above this script.



if [ -f path.sh ]; then . path.sh; fi


#data_list="train test"
data_list="train"

for x in lang lang_test $data_list; do
  mkdir -p data/$x
done

# Copy stuff into its final location:



for x in $data_list; do
  cp data/local/$x.spk2utt data/$x/spk2utt || exit 1;
  cp data/local/$x.utt2spk data/$x/utt2spk || exit 1;
  cp data/local/${x}_wav.scp data/$x/wav.scp || exit 1;
  cp data/local/${x}.txt data/$x/text || exit 1;
  cp data/local/segments* data/$x || exit 1;
done



cp data/local/words.txt data/lang/words.txt
cp data/local/phones.txt data/lang/phones.txt

silphones="SIL SPN NSN"; # This would in general be a space-separated list of all silence phones.  E.g. "sil vn"
# Generate colon-separated lists of silence and non-silence phones.
scripts/silphones.pl data/lang/phones.txt "$silphones" data/lang/silphones.csl \
  data/lang/nonsilphones.csl

ndisambig=`scripts/add_lex_disambig.pl data/local/lexicon.txt data/local/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
scripts/add_disambig.pl data/lang/phones.txt $ndisambig > data/lang_test/phones_disambig.txt


silprob=0.5  # same prob as word
scripts/make_lexicon_fst.pl data/local/lexicon.txt $silprob SIL  | \
  fstcompile --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > data/lang/L.fst

exit 1;

# test lexicon - not ready yet !!!
#scripts/make_lexicon_fst.pl data/local/lexicon_disambig.txt $silprob SIL '#'$ndisambig | \
#   fstcompile --isymbols=data/lang_test/phones_disambig.txt --osymbols=data/lang/words.txt \
#   --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel \
#    > data/lang_test/L_disambig.fst


# G is not ready yet !!!
fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt --keep_isymbols=false \
    --keep_osymbols=false data/local/G.txt > data/lang_test/G.fst

# Checking that G is stochastic [note, it wouldn't be for an Arpa]
fstisstochastic data/lang_test/G.fst || echo Error: G is not stochastic

# Checking that G.fst is determinizable.
fstdeterminize data/lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst | \
   fstdeterminize >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang/L.fst data/lang_test/G.fst | \
   fstisstochastic || echo Error: LG is not stochastic.

# Checking that L_disambig.G is stochastic:
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst | \
   fstisstochastic || echo Error: LG is not stochastic.


## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head


silphonelist=`cat data/lang/silphones.csl | sed 's/:/ /g'`
nonsilphonelist=`cat data/lang/nonsilphones.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > data/lang/topo 

for x in phones.txt words.txt silphones.csl nonsilphones.csl topo; do
   cp data/lang/$x data/lang_test/$x || exit 1;
done

echo SWBD_format_data succeeded.
