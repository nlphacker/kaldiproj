// kwsbin/kws-search.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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
#include "fstext/fstext-utils.h"
#include "lat/kaldi-kws.h"

namespace kaldi {

typedef KwsLexicographicArc Arc;
typedef Arc::Weight Weight;
typedef Arc::StateId StateId;

uint64 EncodeLabel(StateId ilabel,
                   StateId olabel) {
  return (((int64)olabel)<<32)+((int64)ilabel);

}

StateId DecodeLabelUid(uint64 osymbol) {
  // We only need the utterance id
  return ((StateId)(osymbol>>32));
}

class VectorFstToKwsLexicographicFstMapper {
 public:
  typedef fst::StdArc FromArc;
  typedef FromArc::Weight FromWeight;
  typedef KwsLexicographicArc ToArc;
  typedef KwsLexicographicWeight ToWeight;

  VectorFstToKwsLexicographicFstMapper() {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, 
                 arc.olabel,
                 (arc.weight == FromWeight::Zero() ?
                  ToWeight::Zero() :
                  ToWeight(arc.weight.Value(),
                           StdLStdWeight::One())),
                 arc.nextstate);
  }

  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }

  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }

  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS;}

  uint64 Properties(uint64 props) const { return props; }
};

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint32 uint32;
    typedef kaldi::uint64 uint64;
    typedef KwsLexicographicArc Arc;
    typedef Arc::Weight Weight;
    typedef Arc::StateId StateId;

    const char *usage =
        "Search the keywords over the index. This program can be executed parallely, either\n"
        "on the index side or the keywords side; we use a script to combine the final search\n"
        "results. Note that the index archive has a only key \"global\".\n"
        "The output file is in the format:\n"
        "kw utterance_id beg_frame end_frame negated_log_probs\n"
        " e.g.: KW1 1 23 67 0.6074219\n"
        "\n"
        "Usage: kws-search [options]  index-rspecifier keywords-rspecifier results-wspecifier\n"
        " e.g.: kws-search ark:index.idx ark:keywords.fsts ark:results\n";

    ParseOptions po(usage);

    int32 n_best = -1;
    double negative_tolerance = -0.1;
    po.Register("nbest", &n_best, "Return the best n hypotheses.");
    po.Register("negative-tolerance", &negative_tolerance, 
                "The program will die if we get negative score smaller than the tolerance.");
    if (n_best < 0 && n_best != -1) {
      KALDI_ERR << "Bad number for nbest";
      exit (1);
    }

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string index_rspecifier = po.GetArg(1),
        keyword_rspecifier = po.GetOptArg(2),
        result_wspecifier = po.GetOptArg(3);

    RandomAccessTableReader< VectorFstTplHolder<KwsLexicographicArc> > index_reader(index_rspecifier);
    SequentialTableReader<VectorFstHolder> keyword_reader(keyword_rspecifier);
    TableWriter< BasicVectorHolder<double> > result_writer(result_wspecifier);

    // Index has key "global"
    KwsLexicographicFst index = index_reader.Value("global");

    // First we have to remove the disambiguation symbols. But rather than
    // removing them totally, we actually move them from input side to output
    // side, making the output symbol a "combined" symbol of the disambiguation
    // symbols and the utterance id's.
    // Note that in Dogan and Murat's original paper, they simply remove the
    // disambiguation symbol on the input symbol side, which will not allow us
    // to do epsilon removal after composition with the keyword FST. They have
    // to traversal the result FST.
    int32 label_count = 1;
    std::tr1::unordered_map<uint64, uint32> label_encoder;
    std::tr1::unordered_map<uint32, uint64> label_decoder;
    for (StateIterator<KwsLexicographicFst> siter(index); !siter.Done(); siter.Next()) {
      StateId state_id = siter.Value();
      for (MutableArcIterator<KwsLexicographicFst> 
           aiter(&index, state_id); !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        // Skip the non-final arcs
        if (index.Final(arc.nextstate) == Weight::Zero())
          continue;
        // Encode the input and output label of the final arc, and this is the
        // new output label for this arc; set the input label to <epsilon>
        uint64 osymbol = EncodeLabel(arc.ilabel, arc.olabel);
        arc.ilabel = 0;
        if (label_encoder.find(osymbol) == label_encoder.end()) {
          arc.olabel = label_count;
          label_encoder[osymbol] = label_count;
          label_decoder[label_count] = osymbol;
          label_count++;
        } else { 
          arc.olabel = label_encoder[osymbol];
        }
        aiter.SetValue(arc);
      }
    }

    int32 n_done = 0;
    int32 n_fail = 0;
    for (; !keyword_reader.Done(); keyword_reader.Next()) {
      std::string key = keyword_reader.Key();
      VectorFst<StdArc> keyword = keyword_reader.Value();
      keyword_reader.FreeCurrent();

      KwsLexicographicFst kFst;
      KwsLexicographicFst rFst;
      Map(keyword, &kFst, VectorFstToKwsLexicographicFstMapper());
      Compose(kFst, index, &rFst);
      Project(&rFst, PROJECT_OUTPUT);
      Minimize(&rFst);
      ShortestPath(rFst, &rFst, n_best);
      RmEpsilon(&rFst);

      // No result found
      if (rFst.Start() == kNoStateId)
        continue;

      // Got something here
      double score;
      int32 tbeg, tend, uid;
      for (ArcIterator<KwsLexicographicFst> 
           aiter(rFst, rFst.Start()); !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        Weight weight = arc.weight;

        // We're expecting a two-state FST
        if (rFst.Final(arc.nextstate) == Weight::Zero()) {
          KALDI_WARN << "The resulting FST is not a two-state FST for key " << key;
          n_fail++;
          continue;
        }

        uint64 osymbol = label_decoder[arc.olabel];
        uid = (int32)DecodeLabelUid(osymbol);
        tbeg = weight.Value2().Value1().Value();
        tend = weight.Value2().Value2().Value();
        score = weight.Value1().Value();

        if (score < 0) {
          KALDI_ASSERT(score >= negative_tolerance);
          score = 0.0;
        }
        vector<double> result;
        result.push_back(uid);
        result.push_back(tbeg);
        result.push_back(tend);
        result.push_back(score);
        result_writer.Write(key, result);
      }

      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " keywords";
    return (n_done != 0 ? 0 : 1);    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
