# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

export train_cmd="queue.pl -q all.q@a*.clsp.jhu.edu"
export decode_cmd="queue.pl -q all.q@a*.clsp.jhu.edu"
#export train_cmd=run.pl
#export decode_cmd=run.pl



