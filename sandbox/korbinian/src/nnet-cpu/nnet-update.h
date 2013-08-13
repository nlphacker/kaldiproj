// nnet-cpu/nnet-update.h

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

#ifndef KALDI_NNET_CPU_NNET_UPDATE_H_
#define KALDI_NNET_CPU_NNET_UPDATE_H_

#include "nnet-cpu/nnet-nnet.h"
#include "util/table-types.h"

namespace kaldi {
namespace nnet2 {

/* This header provides functionality for sample-by-sample stochastic
   gradient descent and gradient computation with a neural net.
   See also nnet-compute.h which is the same thing but for
   whole utterances.
   This is the inner part of the training code; see nnet-train.h
   which contains a wrapper for this, with functionality for
   automatically keeping the learning rates for each layer updated
   using a heuristic involving validation-set gradients.
*/

// NnetTrainingExample is the input data and corresponding labels (or labels)
// for one frame of input.  In the normal case there will be just one label,
// with a weight of 1.0.  But, for example, in discriminative training there
// might be a mixture of labels with different weights.
struct NnetTrainingExample {
  
  std::vector<std::pair<int32, BaseFloat> > labels;  
  
  Matrix<BaseFloat> input_frames; // The input data-- typically a number of frames
  // (nnet.LeftContext() + 1 + nnet.RightContext()) of raw features, not
  // necessarily contiguous.

  Vector<BaseFloat> spk_info; // The speaker-specific input, if any;
  // a vector of possibly zero length.  We'll append this to each of the
  // input frames.
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};


typedef TableWriter<KaldiObjectHolder<NnetTrainingExample > > NnetTrainingExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetTrainingExample > > SequentialNnetTrainingExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetTrainingExample > > RandomAccessNnetTrainingExampleReader;


/// This function computes the objective function and either updates the model
/// or computes parameter gradients.  Returns the cross-entropy objective
/// function summed over all samples (normalize this by
/// TotalNnetTrainingWeight(examples)).  It is mostly a wrapper for
/// a class NnetUpdater that's defined in nnet-update.cc, but we
/// don't want to expose that complexity at this level.
/// All these examples will be treated as one minibatch.
double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetTrainingExample> &examples,
                  Nnet *nnet_to_update);

/// Returns the total weight summed over all the examples... just a simple
/// utility function.
BaseFloat TotalNnetTrainingWeight(const std::vector<NnetTrainingExample> &egs);

/// Computes objective function over a minibatch.  Returns the *total* weighted
/// objective function over the minibatch.
double ComputeNnetObjf(const Nnet &nnet,
                       const std::vector<NnetTrainingExample> &examples);

/// This version of ComputeNnetObjf breaks up the examples into
/// multiple minibatches to do the computation.
/// Returns the *total* (weighted) objective function.
double ComputeNnetObjf(const Nnet &nnet,                          
                       const std::vector<NnetTrainingExample> &examples,
                       int32 minibatch_size);


/// ComputeNnetGradient is mostly used to compute gradients on validation sets;
/// it divides the example into batches and calls DoBackprop() on each.
/// It returns the *average* objective function per frame.
double ComputeNnetGradient(
    const Nnet &nnet,
    const std::vector<NnetTrainingExample> &examples,
    int32 batch_size,
    Nnet *gradient);


} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET_CPU_NNET_UPDATE_H_
