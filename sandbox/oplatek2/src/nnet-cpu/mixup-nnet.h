// nnet-cpu/mixup-nnet.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET_CPU_MIXUP_NNET_H_
#define KALDI_NNET_CPU_MIXUP_NNET_H_

#include "nnet-cpu/nnet-update.h"
#include "nnet-cpu/nnet-compute.h"
#include "util/parse-options.h"

namespace kaldi {
namespace nnet2 {

/** Configuration class that controls neural net "mixupage" which is actually a
    scaling on the parameters of each of the updatable layers.
 */
struct NnetMixupConfig {
  BaseFloat power;
  BaseFloat min_count;
  int32 num_mixtures;
  BaseFloat perturb_stddev;

  
  NnetMixupConfig(): power(0.25), min_count(1000.0),
                     num_mixtures(-1), perturb_stddev(0.01) { }
  
  void Register(ParseOptions *po) {
    po->Register("power", &power, "Scaling factor used in determining the "
                 "number of mixture components to use for each HMM state "
                 "(or group of HMM states)");
    po->Register("min-count", &min_count, "Minimum count for a quasi-Gaussian, "
                 "enforced while allocating mixtures (obscure parameter).");
    po->Register("num-mixtures", &num_mixtures, "If specified, total number of "
                 "mixture components to mix up to (should be at least the "
                 "#leaves in the system");
    po->Register("perturb-stddev", &perturb_stddev, "Standard deviation used "
                 "when perturbing parameters during mixing up");
  }  
};

/**
  This function does something similar to Gaussian mixture splitting for
  GMMs, except applied to the output layer of the neural network.
  We create additional outputs, which will be summed over using a
  MixtureProbComponent (if one does not already exist, it will be
  added.) 
 */

void MixupNnet(const NnetMixupConfig &mixup_config,
               Nnet *nnet);
  


} // namespace nnet2
} // namespace kaldi

#endif
