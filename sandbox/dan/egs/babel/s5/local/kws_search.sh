#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.


help_message="$(basename $0): do keyword indexing and search.  data-dir is assumed to have
                 kws/ subdirectory that specifies the terms to search for.  Output is in
                 decode-dir/kws/
             Usage:
                 $(basename $0) <lang-dir> <data-dir> <decode-dir>"

# Begin configuration section.  
#acwt=0.0909091
min_lmwt=7
max_lmwt=17
duptime=0.6
cmd=run.pl
model=
skip_scoring=false
skip_optimization=false # true can speed it up if #keywords is small.
max_states=150000
indices_dir=
stage=0
word_ins_penalty=0
extraid=
silence_word=  # specify this if you did to in kws_setup.sh, it's more accurate.
ntrue_scale=1.0
# End configuration section.

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

if [[ "$#" -ne "3" ]] ; then
    echo -e "FATAL: wrong number of script parameters!\n\n"
    printf "$help_message\n\n"
    exit 1;
fi


langdir=$1
datadir=$2
decodedir=$3

if [ -z $extraid ] ; then
  kwsdatadir=$datadir/kws
else
  kwsdatadir=$datadir/${extraid}_kws
fi

if [ -z $extraid ] ; then
  kwsoutdir=$decodedir/kws
else
  kwsoutdir=$decodedir/${extraid}_kws
fi

if [ -z $indices_dir ]; then
  indices_dir=$kwsoutdir
fi

if [ ! -d "$datadir"  ] || [ ! -d "$kwsdatadir" ] ; then
    echo "FATAL: the data directory does not exist"
    exit 1;
fi
if [[ ! -d "$langdir"  ]] ; then
    echo "FATAL: the lang directory does not exist"
    exit 1;
fi
if [[ ! -d "$decodedir"  ]] ; then
    echo "FATAL: the directory with decoded files does not exist"
    exit 1;
fi
if [[ ! -f "$kwsdatadir/ecf.xml"  ]] ; then
    echo "FATAL: the $kwsdatadir does not contain the ecf.xml file"
    exit 1;
fi


duration=`head -1 $kwsdatadir/ecf.xml |\
    grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
    grep -o -E "[0-9]*[\.]*[0-9]*" |\
    perl -e 'while(<>) {print $_/2;}'`

if [ ! -z "$model" ]; then
    model_flags="--model $model"
else
    model_flags=
fi

if [ $stage -le 0 ] ; then
  for lmwt in `seq $min_lmwt $max_lmwt` ; do
      indices=${indices_dir}_$lmwt
      mkdir -p $indices

      acwt=`echo "scale=5; 1/$lmwt" | bc -l | sed "s/^./0./g"` 
      [ ! -z $silence_word ] && silence_opt="--silence-word $silence_word"
      steps/make_index.sh $silence_opt --cmd "$cmd" --acwt $acwt $model_flags\
        --skip-optimization $skip_optimization --max-states $max_states \
        --word-ins-penalty $word_ins_penalty \
        $kwsdatadir $langdir $decodedir $indices  || exit 1
  done
fi

if [ $stage -le 1 ]; then
  for lmwt in `seq $min_lmwt $max_lmwt` ; do
      kwsoutput=${kwsoutdir}_$lmwt
      indices=${indices_dir}_$lmwt
      mkdir -p $kwsoutdir
      steps/search_index.sh --cmd "$cmd" --indices-dir $indices \
        $kwsdatadir $kwsoutput  || exit 1
  done
fi

if [ $stage -le 2 ]; then
  mkdir -p $kwsoutdir
  echo "Writing normalized results"
  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutdir/write_normalized.LMWT.log \
    set -e ';' set -o pipefail ';'\
    cat ${kwsoutdir}_LMWT/result.* \| \
      utils/write_kwslist.pl --Ntrue-scale=$ntrue_scale --flen=0.01 --duration=$duration \
        --segments=$datadir/segments --normalize=true \
        --map-utter=$kwsdatadir/utter_map --digits=3 \
        - - \| local/filter_kwslist.pl $duptime '>' ${kwsoutdir}_LMWT/kwslist.xml || exit 1
fi

if [ $stage -le 3 ]; then
  echo "Writing unnormalized results"
  $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutdir/write_unnormalized.LMWT.log \
    set -e ';' set -o pipefail ';'\
    cat ${kwsoutdir}_LMWT/result.* \| \
        utils/write_kwslist.pl --Ntrue-scale=$ntrue_scale --flen=0.01 --duration=$duration \
          --segments=$datadir/segments --normalize=false \
          --map-utter=$kwsdatadir/utter_map \
          - - \| local/filter_kwslist.pl $duptime '>' ${kwsoutdir}_LMWT/kwslist.unnormalized.xml || exit 1;
fi

if [ -z $extraid ] ; then
  extraid_flags=
else
  extraid_flags="  --extraid ""$extraid"" "
fi

if [ $stage -le 4 ]; then
  if [[ (! -x local/kws_score.sh ) ]] ; then
    echo "Not scoring, because the file local/kws_score.sh is not present"
  elif [[ $skip_scoring == true ]] ; then
    echo "Not scoring, because --skip-scoring true was issued"
  else
    echo "Scoring KWS results"
    $cmd LMWT=$min_lmwt:$max_lmwt $kwsoutdir/scoring.LMWT.log \
       local/kws_score.sh $extraid_flags $datadir ${kwsoutdir}_LMWT || exit 1;
  fi
fi

exit 0
