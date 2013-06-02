// nnet-cpubin/nnet-logprob2.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet-cpu/nnet-randomize.h"
#include "nnet-cpu/nnet-update-parallel.h"
#include "nnet-cpu/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Do the forward computation for a neural net acoustic model, and output\n"
        "matrix of logprobs.  This version of the program outputs to two tables,\n"
        "one table of probabilities without prior division and one table of\n"
        "log-probs with prior division.  It is intended for use in discriminative\n"
        "training.\n"
        "\n"
        "Usage: nnet-logprob2 [options] <model-in> <features-rspecifier> "
        "<probs-wspecifier-not-divided> <logprobs-wspecifier-divided>\n"
        "\n"
        "e.g.: nnet-logprob2 1.nnet \"$feats\" ark:- \"ark:|logprob-to-post ark:- 1.post\" ark:- \\"
        "        | latgen-faster-mapped [args]\n";
    
    std::string spk_vecs_rspecifier, utt2spk_rspecifier;
    bool pad_input = true; // This is not currently configurable.
    
    ParseOptions po(usage);
    
    po.Register("spk-vecs", &spk_vecs_rspecifier, "Rspecifier for a vector that "
                "describes each speaker; only needed if the neural net was "
                "trained this way.");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for map from "
                "utterance to speaker; only relevant in conjunction with the "
                "--spk-vecs option.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        feats_rspecifier = po.GetArg(2),
        prob_wspecifier_nodiv = po.GetArg(3),
        logprob_wspecifier_divided = po.GetArg(4);
        
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    int64 num_done = 0, num_err = 0;

    Vector<BaseFloat> inv_priors(am_nnet.Priors());
    KALDI_ASSERT(inv_priors.Dim() == am_nnet.NumPdfs() &&
                 "Priors in neural network not set up.");
    inv_priors.ApplyPow(-1.0);
    
    SequentialBaseFloatMatrixReader feature_reader(feats_rspecifier);
    // note: spk_vecs_rspecifier and utt2spk_rspecifier may be empty.
    RandomAccessBaseFloatVectorReaderMapped vecs_reader(spk_vecs_rspecifier,
                                                        utt2spk_rspecifier);
    BaseFloatMatrixWriter prob_writer_nodiv(prob_wspecifier_nodiv);
    BaseFloatMatrixWriter logprob_writer_divided(logprob_wspecifier_divided);

    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feats = feature_reader.Value();
      Vector<BaseFloat> spk_vec;
      if (!spk_vecs_rspecifier.empty()) {
        if (!vecs_reader.HasKey(key)) {
          KALDI_ERR << "No speaker vector available for key " << key;
          num_err++;
          continue;
        }
        spk_vec = vecs_reader.Value(key);
      }
      
      Matrix<BaseFloat> log_probs(feats.NumRows(), am_nnet.NumPdfs());
      NnetComputation(am_nnet.GetNnet(), feats, spk_vec, pad_input, &log_probs);
      // at this point "log_probs" contains actual probabilities, not logs.

      // at this point they are probabilities, not log-probs, without prior division.
      prob_writer_nodiv.Write(key, log_probs);
          

      {
        log_probs.MulColsVec(inv_priors); // scales each column by the corresponding element
        // of inv_priors.
        for (int32 i = 0; i < log_probs.NumRows(); i++) {
          SubVector<BaseFloat> frame(log_probs, i);
          BaseFloat p = frame.Sum();
          if (!(p > 0.0)) {
            KALDI_WARN << "Bad sum of probabilities " << p;
          } else {
            frame.Scale(1.0 / p); // re-normalize to sum to one.
          }
        }
      }
      log_probs.ApplyFloor(1.0e-20); // To avoid log of zero which leads to NaN.
      log_probs.ApplyLog();
      
      logprob_writer_divided.Write(key, log_probs);
      num_done++;
    }
    
    KALDI_LOG << "Finished computing neural net log-probs, processed "
              << num_done << " utterances, " << num_err << " with errors.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


