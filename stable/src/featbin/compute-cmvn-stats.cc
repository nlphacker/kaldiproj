// featbin/compute-cmvn-stats.cc

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
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Compute cepstral mean and variance normalization statistics\n"
        "If wspecifier provided: per-utterance by default, or per-speaker if\n"
        "spk2utt option provided; if wxfilename: global\n"
        "Usage: compute-cmvn-stats  [options] feats-rspecifier (stats-wspecifier|stats-wxfilename)\n";
    
    ParseOptions po(usage);
    std::string spk2utt_rspecifier;
    bool binary = true;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to utterance-list map");
    po.Register("binary", &binary, "write in binary mode (applies only to global CMN/CVN)");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0, num_err = 0;
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier_or_wxfilename = po.GetArg(2);

    if (ClassifyWspecifier(wspecifier_or_wxfilename, NULL, NULL, NULL)
        != kNoWspecifier) { // writing to a Table: per-speaker or per-utt CMN/CVN.
      std::string wspecifier = wspecifier_or_wxfilename;

      DoubleMatrixWriter writer(wspecifier);

      if (spk2utt_rspecifier != "") {
        SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
        RandomAccessBaseFloatMatrixReader feat_reader(rspecifier);
        for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
          std::string spk = spk2utt_reader.Key();
          const std::vector<std::string> &uttlist = spk2utt_reader.Value();
          bool is_init = false;
          Matrix<double> stats;
          for (size_t i = 0; i < uttlist.size(); i++) {
            std::string utt = uttlist[i];
            if (!feat_reader.HasKey(utt))
              KALDI_WARN << "Did not find features for utterance " << utt;
            else {
              const Matrix<BaseFloat> &feats = feat_reader.Value(utt);
              if (!is_init) {
                InitCmvnStats(feats.NumCols(), &stats);
                is_init = true;
              }
              AccCmvnStats(feats, NULL, &stats);
            }
          }
          if (stats.NumRows() == 0) {
            num_err++;
            KALDI_WARN << "No stats accumulated for speaker " << spk;
          } else {
            num_done++;
            writer.Write(spk, stats);
          }
        }
      } else {  // per-utterance normalization
        SequentialBaseFloatMatrixReader feat_reader(rspecifier);
        for (; !feat_reader.Done(); feat_reader.Next()) {
          Matrix<double> stats;
          const Matrix<BaseFloat> &feats = feat_reader.Value();
          InitCmvnStats(feats.NumCols(), &stats);
          AccCmvnStats(feats, NULL, &stats);
          writer.Write(feat_reader.Key(), stats);
          num_done++;
        }
      }
    } else { // accumulate global stats
      if (spk2utt_rspecifier != "")
        KALDI_ERR << "--spk2utt option not compatible with wxfilename as output "
                   << "(did you forget ark:?)";
      std::string wxfilename = wspecifier_or_wxfilename;
      bool is_init = false;
      Matrix<double> stats;
      SequentialBaseFloatMatrixReader feat_reader(rspecifier);
      for (; !feat_reader.Done(); feat_reader.Next()) {
        const Matrix<BaseFloat> &feats = feat_reader.Value();
        if (!is_init) {
          InitCmvnStats(feats.NumCols(), &stats);
          is_init = true;
        }
        AccCmvnStats(feats, NULL, &stats);
      }
      Matrix<float> fstats(stats);
      Output ko(wxfilename, binary);
      fstats.Write(ko.Stream(), binary);
      num_done++;
    }
    KALDI_LOG << "Done accumulating CMVN stats for " << num_done
              << " separate entities (e.g. speakers); " << num_err
              << " had errors.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


