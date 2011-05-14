// feat/feature-plp.h

// Copyright 2009-2011  Petr Motlicek, Karel Vesely

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

#ifndef KALDI_FEAT_FEATURE_PLP_H_
#define KALDI_FEAT_FEATURE_PLP_H_



#include <string>
#include "feat/feature-functions.h"
#include "util/parse-options.h"
#include "matrix/kaldi-matrix-inl.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{



/// PlpOptions contains basic options for computing PLP features.
/// It only includes things that can be done in a "stateless" way, i.e.
/// it does not include energy max-normalization.
/// It does not include delta computation.
struct PlpOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  int32 lpc_order;
  int32 num_ceps;  // num cepstra including zero
  bool use_energy;  // use energy; else C0
  BaseFloat energy_floor;
  bool raw_energy;  // compute energy before preemphasis and hamming window (else after)
  BaseFloat compress_factor;
  int32 cepstral_lifter;
  BaseFloat cepstral_scale;

  bool htk_compat;  // if true, put energy/C0 last and introduce a factor of sqrt(2)

  PlpOptions(): mel_opts(23),  // default number of mel-banks for the PLP computation;
                // maybe should use 15 for 8khz.
                lpc_order(12),
                num_ceps(13),
                use_energy(true),
                energy_floor(0.0),  // not in log scale: a small value e.g. 1.0e-10
                raw_energy(true),
                compress_factor(0.33333),
                cepstral_lifter(22),
                cepstral_scale(1.0),
                htk_compat(false) { }

  void Register(ParseOptions *po) {
    frame_opts.Register(po);
    mel_opts.Register(po);
    po->Register("lpc-order", &lpc_order, "Order of LPC analysis in PLP computation");
    po->Register("num-ceps", &num_ceps, "Number of cepstra in PLP computation (including C0)");
    po->Register("use-energy", &use_energy, "Use energy (not C0) in MFCC computation");
    po->Register("energy-floor", &energy_floor, "Floor on energy (absolute, not relative) in PLP computation");
    po->Register("raw-energy", &raw_energy, "If true, compute energy (if using energy) before Hamming window and preemphasis");
    po->Register("compress-factor", &compress_factor, "Compression factor in PLP computation");
    po->Register("cepstral-lifter", &cepstral_lifter, "Constant that controls scaling of PLPs");
    po->Register("cepstral-scale", &cepstral_scale, "Scaling constant in PLP computation");
    po->Register("htk-compat", &htk_compat, "If true, put energy or C0 last and put factor of sqrt(2) on C0.");
  }
};


/// Class for computing PLP features.  See \ref feat_plp where
/// documentation will eventually be added.
class Plp {
 public:
  Plp(const PlpOptions &opts);

  ~Plp();

  void Compute(const VectorBase<BaseFloat> &wave,
               BaseFloat vtln_warp,
               Matrix<BaseFloat> *output,
               Vector<BaseFloat> *wave_remainder = NULL);

 private:
  const MelBanks *GetMelBanks(BaseFloat vtln_warp);
  const Vector<BaseFloat> *GetEqualLoudness(BaseFloat vtln_warp);
  PlpOptions opts_;
  Vector<BaseFloat> lifter_coeffs_;
  Matrix<BaseFloat> idft_bases_;
  BaseFloat log_energy_floor_;
  std::map<BaseFloat, MelBanks*> mel_banks_;  // BaseFloat is VTLN coefficient.
  std::map<BaseFloat, Vector<BaseFloat>* > equal_loudness_;
  FeatureWindowFunction feature_window_function_;
  SplitRadixRealFft<BaseFloat> *srfft_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Plp);
};

/// @} End of "addtogroup feat"

}// namespace kaldi


#endif
