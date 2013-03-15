#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Yenda Trmal)
# Apache 2.0.

# Begin configuration section.
# case_insensitive=true
extraid=
# End configuration section.

help_message="$0: create subset of the input directory (specified as the first directory).
                 The subset is specified by the second parameter.
                 The directory in which the subset should be created is the third parameter
             Example:
                 $0 <source-corpus-dir> <subset-descriptor-list-file> <target-corpus-subset-dir>"

echo $0 $@
[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 2 ]; then
    printf "FATAL: incorrect number of variables given to the script\n\n"
    printf "$help_message\n"
    exit 1;
fi

if [ -z $extraid ] ; then
  kwsdatadir=$1/kws
else
  kwsdatadir=$1/kws_${extraid}
fi
kwsoutputdir="$2/"

if [[ ! -d "$kwsdatadir" ]] ; then
    echo "FATAL: the KWS input data directory does not exist!"
    exit 1;
fi

for file in ecf.xml rttm kwlist.xml ; do
    if [[ ! -f "$kwsdatadir/$file" ]] ; then
        echo "FATAL: file $kwsdatadir/$file does not exist!"
        exit 1;
    fi
done

echo KWSEval -e $kwsdatadir/ecf.xml -r $kwsdatadir/rttm -t $kwsdatadir/kwlist.xml \
    -s $kwsoutputdir/kwslist.xml -c -o -b -d -f $kwsoutputdir

KWSEval -e $kwsdatadir/ecf.xml -r $kwsdatadir/rttm -t $kwsdatadir/kwlist.xml \
    -s $kwsoutputdir/kwslist.xml -c -o -b -d -f $kwsoutputdir || exit 1;

exit 0;


