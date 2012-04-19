// fgmmbin/fgmm-gselect.cc

// Copyright 2009-2011   Saarland University;  Microsoft Corporation

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
#include "gmm/full-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using std::vector;
    typedef kaldi::int32 int32;
    const char *usage =
        "Precompute Gaussian indices for pruning\n"
        " (e.g. in training UBMs, SGMMs, tied-mixture systems)\n"
        " For each frame, gives a list of the n best Gaussian indices,\n"
        " sorted from best to worst.\n"
        "Usage: \n"
        " fgmm-gselect [options] <model-in> <feature-rspecifier> <gselect-wspecifier>\n"
        "The --gselect option (which takes an rspecifier) limits selection to a subset\n"
        "of indices:\n"
        "e.g.: fgmm-gselect \"--gselect=ark:gunzip -c bigger.gselect.gz|\" --n=20 1.gmm \"ark:feature-command |\" \"ark,t:|gzip -c >1.gselect.gz\n";
    
    ParseOptions po(usage);
    int32 num_gselect = 50;
    std::string gselect_rspecifier;    
    std::string likelihood_wspecifier;
    po.Register("n", &num_gselect, "Number of Gaussians to keep per frame\n");
    po.Register("write-likes", &likelihood_wspecifier, "Wspecifier for likelihoods per "
                "utterance");
    po.Register("gselect", &gselect_rspecifier, "rspecifier for gselect objects "
                "to limit the search to");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        gselect_wspecifier = po.GetArg(3);
    
    FullGmm fgmm;
    ReadKaldiObject(model_filename, &fgmm);
    KALDI_ASSERT(num_gselect > 0);
    int32 num_gauss = fgmm.NumGauss();
    KALDI_ASSERT(num_gauss);
    if (num_gselect > num_gauss) {
      KALDI_WARN << "You asked for " << num_gselect << " Gaussians but GMM "
                 << "only has " << num_gauss << ", returning this many. "
                 << "Note: this means the Gaussian selection is pointless.";
      num_gselect = num_gauss;
    }
    
    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;
    bool warned_size = false;
    
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorVectorWriter gselect_writer(gselect_wspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier); // may be ""
    BaseFloatWriter likelihood_writer(likelihood_wspecifier); // may be ""

    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      int32 tot_t_this_file = 0; double tot_like_this_file = 0;
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      vector<vector<int32> > gselect(mat.NumRows());
      tot_t_this_file += mat.NumRows();
      if(gselect_rspecifier != "") { // Limit Gaussians to preselected group...
        if(!gselect_reader.HasKey(utt)) {
          KALDI_WARN << "No gselect information for utterance " << utt;
          num_err++;
          continue;
        }
        const vector<vector<int32> > &preselect = gselect_reader.Value(utt);
        if (preselect.size() != static_cast<size_t>(mat.NumRows())) {
          KALDI_WARN << "Input gselect for utterance " << utt << " has wrong size "
                     << preselect.size() << " vs. " << mat.NumRows();
          num_err++;
          continue;
        }
        for (int32 i = 0; i < mat.NumRows(); i++) {
          int32 this_num_gselect = num_gselect;
          int32 preselect_sz = preselect[i].size();
          if (preselect_sz < num_gselect && !warned_size) {
            this_num_gselect = preselect_sz;
            warned_size = true;
            KALDI_WARN << "Preselect size is less than final size, "
                       << preselect_sz << " < " <<  num_gselect
                       << " [won't warn again]";
          }
          Vector<BaseFloat> loglikes(preselect_sz);
          fgmm.LogLikelihoodsPreselect(mat.Row(i), preselect[i], &loglikes);
          Vector<BaseFloat> loglikes_copy(loglikes);
          BaseFloat *ptr = loglikes_copy.Data();
          std::nth_element(ptr, ptr+preselect_sz-this_num_gselect,
                           ptr+preselect_sz);
          BaseFloat thresh = ptr[preselect_sz-this_num_gselect];

          BaseFloat loglike = -1.0e+10;
          // we want the output sorted from best likelihood to worse
          // (so we can prune further without the model)...
          std::vector<std::pair<BaseFloat,int32> > pairs;
          for (int32 p = 0; p < preselect_sz; p++) {
            if (loglikes(p) >= thresh) {
              pairs.push_back(std::make_pair(loglikes(p), preselect[i][p]));
            }
          }
          std::sort(pairs.begin(), pairs.end(),
                    std::greater<std::pair<BaseFloat,int32> >());
          for (int32 j = 0;
               j < this_num_gselect && j < static_cast<int32>(pairs.size());
               j++) {
            gselect[i].push_back(pairs[j].second);
            loglike = LogAdd(loglike, pairs[j].first);
          }
          
          tot_like_this_file += loglike;
          if (gselect[i].empty())
            KALDI_WARN << "Selected no Gaussians on frame " << i
                       << " of utterance " << utt << " [NaNs?]";
        }
      } else { // No "preselect" [i.e. no existing gselect]: simple case.
        Vector<BaseFloat> loglikes(num_gauss);
        for (int32 i = 0; i < mat.NumRows(); i++) {
          fgmm.LogLikelihoods(mat.Row(i), &loglikes);
          Vector<BaseFloat> loglikes_copy(loglikes);
          BaseFloat *ptr = loglikes_copy.Data();
          std::nth_element(ptr, ptr+num_gauss-num_gselect,
                           ptr+num_gauss);
          BaseFloat thresh = ptr[num_gauss-num_gselect];
          
          BaseFloat loglike = -1.0e+10;
          std::vector<std::pair<BaseFloat,int32> > pairs;
          for (int32 p = 0; p < num_gauss; p++) {
            if (loglikes(p) >= thresh) {
              pairs.push_back(std::make_pair(loglikes(p), p));
            }
          }
          std::sort(pairs.begin(), pairs.end(),
                    std::greater<std::pair<BaseFloat,int32> >());
          for (int32 j = 0;
               j < num_gselect && j < static_cast<int32>(pairs.size());
               j++) {
            gselect[i].push_back(pairs[j].second);
            loglike = LogAdd(loglike, pairs[j].first);
          }
          tot_like_this_file += loglike;
          if (gselect[i].empty())
            KALDI_WARN << "Selected no Gaussians on frame " << i
                       << " of utterance " << utt << " [NaNs?]";
        }
      }
      
      gselect_writer.Write(utt, gselect);
      if (num_done % 10 == 0)
        KALDI_LOG << "For " << num_done << "'th file, average UBM likelihood over "
                  << tot_t_this_file << " frames is "
                  << (tot_like_this_file/tot_t_this_file);
      tot_t += tot_t_this_file;
      tot_like += tot_like_this_file;
      
      if(likelihood_wspecifier != "")
        likelihood_writer.Write(utt, tot_like_this_file);
      num_done++;
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors, average UBM log-likelihood is "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";
    
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


