// nnet-cpu/decodable-am-nnet.h

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

#ifndef KALDI_NNET_CPU_DECODABLE_AM_NNET_H_
#define KALDI_NNET_CPU_DECODABLE_AM_NNET_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet-cpu/am-nnet.h"
#include "nnet-cpu/nnet-compute.h"

namespace kaldi {
namespace nnet2 {

/// DecodableAmNnet is a decodable object that decodes
/// with a neural net acoustic model of type AmNnet.

class DecodableAmNnet: public DecodableInterface {
 public:
  DecodableAmNnet(const TransitionModel &trans_model,
                  const AmNnet &am_nnet,
                  const MatrixBase<BaseFloat> &feats,
                  const VectorBase<BaseFloat> &spk_info,
                  bool pad_input = true, // if !pad_input, the NumIndices()
                  // will be < feats.NumRows().
                  BaseFloat prob_scale = 1.0):
      trans_model_(trans_model) {
    // Note: we could make this more memory-efficient by doing the
    // computation in smaller chunks than the whole utterance, and not
    // storing the whole thing.  We'll leave this for later.
    log_probs_.Resize(feats.NumRows(), trans_model.NumPdfs());
    // the following function is declared in nnet-compute.h
    NnetComputation(am_nnet.GetNnet(), feats, spk_info, pad_input, &log_probs_);
    log_probs_.ApplyFloor(1.0e-20); // Avoid log of zero which leads to NaN.
    log_probs_.ApplyLog();
    Vector<BaseFloat> priors(am_nnet.Priors());
    KALDI_ASSERT(priors.Dim() == trans_model.NumPdfs() &&
                 "Priors in neural network not set up.");
    priors.ApplyLog();
    // subtract log-prior (divide by prior)
    log_probs_.AddVecToRows(-1.0, priors);
    // apply probability scale.
    log_probs_.Scale(prob_scale);
  }

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id) {
    return log_probs_(frame,
                      trans_model_.TransitionIdToPdf(transition_id));
  }

  int32 NumFrames() { return log_probs_.NumRows(); }
  
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }
  
  virtual bool IsLastFrame(int32 frame) {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

 protected:
  const TransitionModel &trans_model_;
  Matrix<BaseFloat> log_probs_; // actually not really probabilities, since we divide
  // by the prior -> they won't sum to one.

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnet);
};

/// This version of DecodableAmNnet is intended for a version of the decoder
/// that processes different utterances with multiple threads.  It needs to do
/// the computation in a different place than the initializer, since the
/// initializer gets called in the main thread of the program.

class DecodableAmNnetParallel: public DecodableInterface {
 public:
  DecodableAmNnetParallel(
      const TransitionModel &trans_model,
      const AmNnet &am_nnet,
      const Matrix<BaseFloat> *feats,
      const Vector<BaseFloat> *spk_info,
      bool pad_input = true,
      BaseFloat prob_scale = 1.0):
      trans_model_(trans_model), am_nnet_(am_nnet), feats_(feats),
      spk_info_(spk_info), pad_input_(pad_input), prob_scale_(prob_scale) {
    KALDI_ASSERT(feats_ != NULL && spk_info_ != NULL);
  }

  void Compute() {
    log_probs_.Resize(feats_->NumRows(), trans_model_.NumPdfs());
    // the following function is declared in nnet-compute.h
    NnetComputation(am_nnet_.GetNnet(), *feats_,
                    *spk_info_, pad_input_, &log_probs_);
    log_probs_.ApplyFloor(1.0e-20); // Avoid log of zero which leads to NaN.
    log_probs_.ApplyLog();
    Vector<BaseFloat> priors(am_nnet_.Priors());
    KALDI_ASSERT(priors.Dim() == trans_model_.NumPdfs() &&
                 "Priors in neural network not set up.");
    priors.ApplyLog();
    // subtract log-prior (divide by prior)
    log_probs_.AddVecToRows(-1.0, priors);
    // apply probability scale.
    log_probs_.Scale(prob_scale_);
    delete feats_;
    feats_ = NULL;
    delete spk_info_;
    spk_info_ = NULL;
  }

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id) {
    if (feats_) Compute(); // this function sets feats_ to NULL.
    return log_probs_(frame,
                      trans_model_.TransitionIdToPdf(transition_id));
  }

  int32 NumFrames() {
    if (feats_) Compute();
    return log_probs_.NumRows();
  }
  
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }
  
  virtual bool IsLastFrame(int32 frame) {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }
  ~DecodableAmNnetParallel() {
    if (feats_) delete feats_;
    if (spk_info_) delete spk_info_;
  }
 protected:
  const TransitionModel &trans_model_;
  const AmNnet &am_nnet_;
  Matrix<BaseFloat> log_probs_; // actually not really probabilities, since we divide
  // by the prior -> they won't sum to one.
  const Matrix<BaseFloat> *feats_;
  const Vector<BaseFloat> *spk_info_;
  bool pad_input_;
  BaseFloat prob_scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetParallel);
};




  
} // namespace nnet2
} // namespace kaldi

#endif  // KALDI_NNET_CPU_DECODABLE_AM_NNET_H_
