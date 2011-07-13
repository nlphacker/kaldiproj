// gmmbin/gmm-acc-stats-ali.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/estimate-am-diag-gmm.h"




int main(int argc, char *argv[])
{
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate stats for GMM training.\n"
        "Usage:  gmm-acc-stats-ali [options] <model-in> <feature-rspecifier> <alignments-rspecifier> <stats-out>\n"
        "e.g.: \n"
        " gmm-acc-stats-ali 1.mdl 1.ali scp:train.scp ark:1.ali 1.acc\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_gmm.Read(is.Stream(), binary);
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);
    MlEstimateAmDiagGmm gmm_accs;
    gmm_accs.InitAccumulators(am_gmm, kGmmAll);

    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!alignments_reader.HasKey(key)) {
        num_no_alignment++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(key);

        if (alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0;

        for (size_t i = 0; i < alignment.size(); i++) {
          int32 tid = alignment[i],  // transition identifier.
              pdf_id = trans_model.TransitionIdToPdf(tid);
          trans_model.Accumulate(1.0, tid, &transition_accs);
          tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, mat.Row(i), pdf_id, 1.0);
        }
        KALDI_LOG << "Average like for this file is "
                  << (tot_like_this_file/alignment.size()) << " over "
                  << alignment.size() <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += alignment.size();
        if (num_done % 10 == 0)
          KALDI_LOG << "Avg like per frame so far is "
                    << (tot_like/tot_t);
      }
    }
    KALDI_LOG << "Done " << num_done << " files, " << num_no_alignment
              << " with no alignments, " << num_other_error
              << " with other errors.";

    KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";
    
    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


