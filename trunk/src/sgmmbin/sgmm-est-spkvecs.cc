// sgmmbin/sgmm-est-spkvecs.cc

// Copyright 2009-2011  Arnab Ghoshal (Saarland University), Microsoft Corporation

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

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "sgmm/am-sgmm.h"
#include "sgmm/estimate-am-sgmm.h"
#include "hmm/transition-model.h"

namespace kaldi {

void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Posterior &post,
                            const TransitionModel &trans_model,
                            const AmSgmm &am_sgmm,
                            const SgmmGselectConfig &gselect_opts,
                            const vector< vector<int32> > &gselect,
                            const SgmmPerSpkDerivedVars &spk_vars,
                            MleSgmmSpeakerAccs *spk_stats) {
  kaldi::SgmmPerFrameDerivedVars per_frame_vars;

  for (size_t i = 0; i < post.size(); i++) {
    std::vector<int32> this_gselect;
    if (!gselect.empty())
      this_gselect = gselect[i];
    else
      am_sgmm.GaussianSelection(gselect_opts, feats.Row(i), &this_gselect);
    am_sgmm.ComputePerFrameVars(feats.Row(i), this_gselect, spk_vars, 0.0, &per_frame_vars);

    for (size_t j = 0; j < post[i].size(); j++) {
      int32 pdf_id = trans_model.TransitionIdToPdf(post[i][j].first);
      spk_stats->Accumulate(am_sgmm, per_frame_vars, pdf_id, post[i][j].second);
    }
  }
}

}  // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate SGMM speaker vectors, either per utterance or for the "
        "supplied set of speakers (with spk2utt option).\n"
        "Reads Gaussian-level posteriors. Writes to a table of vectors.\n"
        "Usage: sgmm-est-spkvecs [options] <model-in> <feature-rspecifier> "
        "<post-rspecifier> <vecs-wspecifier>\n";

    ParseOptions po(usage);
    string gselect_rspecifier, spk2utt_rspecifier, spkvecs_rspecifier;
    BaseFloat min_count = 100;
    BaseFloat rand_prune = 1.0e-05;
    SgmmGselectConfig gselect_opts;

    gselect_opts.Register(&po);
    po.Register("gselect", &gselect_rspecifier,
        "File to read precomputed per-frame Gaussian indices from.");
    po.Register("spk2utt", &spk2utt_rspecifier,
        "File to read speaker to utterance-list map from.");
    po.Register("spkvec-min-count", &min_count,
        "Minimum count needed to estimate speaker vectors");
    po.Register("rand-prune", &rand_prune, "Pruning threshold for posteriors");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors to use during aligment (rspecifier)");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    string model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        post_rspecifier = po.GetArg(3),
        vecs_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmSgmm am_sgmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }
    MleSgmmSpeakerAccs spk_stats(am_sgmm);

    RandomAccessPosteriorReader post_reader(post_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader;
    if (!gselect_rspecifier.empty())
      if (!gselect_reader.Open(gselect_rspecifier))
        KALDI_ERR << "Cannot open stream to read gaussian-selection indices";

    RandomAccessBaseFloatVectorReader spkvecs_reader;
    if (!spkvecs_rspecifier.empty())
      if (!spkvecs_reader.Open(spkvecs_rspecifier))
        KALDI_ERR << "Cannot read speaker vectors.";

    BaseFloatVectorWriter vecs_writer(vecs_wspecifier);

    double tot_impr = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_no_post = 0, num_other_error = 0;
    std::vector<std::vector<int32> > empty_gselect;

    if (!spk2utt_rspecifier.empty()) {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        spk_stats.Clear();
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(spk)) {
            spk_vars.v_s = spkvecs_reader.Value(spk);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << spk;
          }
        }  // else spk_vars is "empty"

        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            continue;
          }
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_post++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &post = post_reader.Value(utt);
          if (static_cast<int32>(post.size()) != feats.NumRows()) {
            KALDI_WARN << "Posterior vector has wrong size " << (post.size())
                       << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }
          bool has_gselect = false;
          if (gselect_reader.IsOpen()) {
            has_gselect = gselect_reader.HasKey(utt)
                          && gselect_reader.Value(utt).size() == feats.NumRows();
            if (!has_gselect)
              KALDI_WARN << "No Gaussian-selection info available for utterance "
                         << utt << " (or wrong size)";
          }
          const std::vector<std::vector<int32> > *gselect =
              (has_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

          AccumulateForUtterance(feats, post, trans_model, am_sgmm, gselect_opts, *gselect, spk_vars, &spk_stats);
          num_done++;
        }  // end looping over all utterances of the current speaker

        BaseFloat impr, spk_tot_t;
        {  // Compute the spk_vec and write it out.
          Vector<BaseFloat> spk_vec(am_sgmm.SpkSpaceDim(), kSetZero);
          if (spk_vars.v_s.Dim() != 0) spk_vec.CopyFromVec(spk_vars.v_s);
          spk_stats.Update(min_count, &spk_vec, &impr, &spk_tot_t);
          vecs_writer.Write(spk, spk_vec);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from speaker vector is "
                  << (impr/spk_tot_t) << ", over " << spk_tot_t << " frames.\n";
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        if (!post_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find posts for utterance "
                     << utt;
          num_no_post++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.v_s = spkvecs_reader.Value(utt);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
          }
        }  // else spk_vars is "empty"
        const Posterior &post = post_reader.Value(utt);

        if (static_cast<int32>(post.size()) != feats.NumRows()) {
          KALDI_WARN << "Posterior has wrong size " << (post.size())
              << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }
        num_done++;

        spk_stats.Clear();
        bool has_gselect = false;
        if (gselect_reader.IsOpen()) {
          has_gselect = gselect_reader.HasKey(utt)
                        && gselect_reader.Value(utt).size() == feats.NumRows();
          if (!has_gselect)
            KALDI_WARN << "No Gaussian-selection info available for utterance "
                       << utt << " (or wrong size)";
        }
        const std::vector<std::vector<int32> > *gselect =
            (has_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

        AccumulateForUtterance(feats, post, trans_model, am_sgmm, gselect_opts, *gselect, spk_vars, &spk_stats);

        BaseFloat impr, utt_tot_t;
        {  // Compute the spk_vec and write it out.
          Vector<BaseFloat> spk_vec(am_sgmm.SpkSpaceDim(), kSetZero);
          if (spk_vars.v_s.Dim() != 0) spk_vec.CopyFromVec(spk_vars.v_s);
          spk_stats.Update(min_count, &spk_vec, &impr, &utt_tot_t);
          vecs_writer.Write(utt, spk_vec);
        }
        KALDI_LOG << "For utterance " << utt << ", auxf-impr from speaker vectors is "
                  << (impr/utt_tot_t) << ", over " << utt_tot_t << " frames.\n";
        tot_impr += impr;
        tot_t += utt_tot_t;
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_post
              << " with no posts, " << num_other_error << " with other errors.";
    KALDI_LOG << "Num frames " << tot_t << ", auxf impr per frame is "
              << (tot_impr / tot_t);
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

