#!/bin/bash
#
# Copyright  2014 Nickolay V. Shmyrev 
#            2014 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0
#

dir=data/local/dict
mkdir -p $dir

srcdict=db/TEDLIUM_release1/TEDLIUM.150K.dic

[ ! -r $srcdict ] && echo "Missing $srcdict" && exit 1
[ ! -r db/extra.dic ] && echo "Missing db/extra.dic" && exit 1

# Join dicts and fix some troubles
cat $srcdict db/extra.dic | LANG= LC_ALL= sort | sed 's:([0-9])::g' |
  grep -vw "ei" |
  grep -vw "erj" |
  grep -v "text2pho.sh" > $dir/lexicon_words.txt 

cat $dir/lexicon_words.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' | \
  grep -v SIL | sort > $dir/nonsilence_phones.txt  

( echo SIL; echo BRH; echo CGH; echo NSN ; echo SMK; echo UM; echo UHH ) > $dir/silence_phones.txt

echo SIL > $dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone.
echo -n >$dir/extra_questions.txt

# Add to the lexicon the silences, noises etc.
(echo '!SIL SIL'; echo '[BREATH] BRH'; echo '[NOISE] NSN'; echo '[COUGH] CGH';
 echo '[SMACK] SMK'; echo '[UM] UM'; echo '[UH] UHH'
 echo '<UNK> NSN' ) | \
 cat - $dir/lexicon_words.txt  > $dir/lexicon.txt

# Check that the dict dir is okay!
utils/validate_dict_dir.pl $dir || exit 1
