// online2/online-nnet2-decoding.h

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
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


#ifndef KALDI_ONLINE2_ONLINE_NNET2_DECODING_H_
#define KALDI_ONLINE2_ONLINE_NNET2_DECODING_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "nnet2/online-nnet2-decodable.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-endpoint.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

namespace kaldi {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{





// This configuration class contains the configuration classes needed to create
// the class SingleUtteranceNnet2Decoder.  The actual command line program
// requires other configs that it creates separately, and which are not included
// here: namely, OnlineNnet2FeaturePipelineConfig and OnlineEndpointConfig.
struct OnlineNnet2DecodingConfig {
  
  LatticeFasterDecoderConfig faster_decoder_opts;
  nnet2::DecodableNnet2OnlineOptions decodable_opts;
  
  OnlineNnet2DecodingConfig() {  decodable_opts.acoustic_scale = 0.1; }
  
  void Register(OptionsItf *po) {
    faster_decoder_opts.Register(po);
    decodable_opts.Register(po);
  }
};

/**
   You will instantiate this class when you want to decode a single
   utterance using the online-decoding setup for neural nets.
*/
class SingleUtteranceNnet2Decoder {
 public:
  // Constructor.  The feature_pipeline_ pointer is not owned in this
  // class, it's owned externally.
  SingleUtteranceNnet2Decoder(const OnlineNnet2DecodingConfig &config,
                              const TransitionModel &tmodel,
                              const nnet2::AmNnet &model,
                              const fst::Fst<fst::StdArc> &fst,
                              OnlineNnet2FeaturePipeline *feature_pipeline);
  
  /// advance the decoding as far as we can.
  void AdvanceDecoding();

  int32 NumFramesDecoded() const;
  
  /// Gets the lattice.  The output lattice has any acoustic scaling in it
  /// (which will typically be desirable in an online-decoding context); if you
  /// want an un-scaled lattice, scale it using ScaleLattice() with the inverse
  /// of the acoustic weight.  "end_of_utterance" will be true if you want the
  /// final-probs to be included.
  void GetLattice(bool end_of_utterance,
                  CompactLattice *clat) const;

  /// This function outputs to "final_relative_cost", if non-NULL, a number >= 0
  /// that will be close to zero if the final-probs were close to the best probs
  /// active on the final frame.  (the output to final_relative_cost is based on
  /// the first-pass decoding).  If it's close to zero (e.g. < 5, as a guess),
  /// it means you reached the end of the grammar with good probability, which
  /// can be taken as a good sign that the input was OK.
  BaseFloat FinalRelativeCost() { return decoder_.FinalRelativeCost(); }

  /// This function calls EndpointDetected from online-endpoint.h,
  /// with the required arguments.
  bool EndpointDetected(const OnlineEndpointConfig &config);

  ~SingleUtteranceNnet2Decoder() { }
 private:

  OnlineNnet2DecodingConfig config_;

  OnlineNnet2FeaturePipeline *feature_pipeline_;

  const TransitionModel &tmodel_;
  
  nnet2::DecodableNnet2Online decodable_;
  
  LatticeFasterOnlineDecoder decoder_;
  
};

  
/// @} End of "addtogroup onlinedecoding"

}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_NNET2_DECODING_H_
