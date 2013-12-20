#!/bin/bash

# You don't have to run all these.  
# you can pick and choose.  Look at the RESULTS file..


use_gpu=true  

if $use_gpu; then
  # This example runs on top of "raw-fMLLR" features:
  # We don't have a GPU version of this script.
  #local/nnet2/run_4a_gpu.sh

  # This one is on top of filter-bank features, with only CMN.
  local/nnet2/run_4b_gpu.sh

  # This one is on top of 40-dim + fMLLR features
  local/nnet2/run_4c_gpu.sh

  # This is discriminative training on top of 4c.
  local/nnet2/run_5c_gpu.sh
else
  # This example runs on top of "raw-fMLLR" features:
  local/nnet2/run_4a.sh

  # This one is on top of filter-bank features, with only CMN.
  local/nnet2/run_4b.sh

  # This one is on top of 40-dim + fMLLR features
  local/nnet2/run_4c.sh

  # This is discriminative training on top of 4c.
  local/nnet2/run_5c.sh
fi
  
