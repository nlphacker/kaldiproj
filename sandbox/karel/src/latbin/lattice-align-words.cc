// latbin/lattice-align-words.cc

// Copyright 2012  Daniel Povey

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
#include "lat/kaldi-lattice.h"
#include "lat/word-align-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::StdArc;
    using kaldi::int32;

    const char *usage =
        "Convert lattices so that the arcs in the CompactLattice format correspond with\n"
        "words (i.e. aligned with word boundaries).\n"
        "Usage: lattice-align-words [options] <word-boundary-file> <model> <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-align-words  --silence-label=4320 --partial-word-label=4324 \\\n"
        "   data/lang/phones/word_boundary.int final.mdl ark:1.lats ark:aligned.lats\n"
        "Note: word-boundary file has format (on each line):\n"
        "<integer-phone-id> [begin|end|singleton|internal|nonword]\n";
    
    ParseOptions po(usage);
    bool output_if_error = true;
    bool do_test = false;
    
    po.Register("output-error-lats", &output_if_error, "Output lattices that aligned "
                "with errors (e.g. due to force-out");
    po.Register("test", &do_test, "Test the algorithm while running it.");
    
    WordBoundaryInfoNewOpts opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        word_boundary_rxfilename = po.GetArg(1),
        model_rxfilename = po.GetArg(2),
        lats_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    TransitionModel tmodel;
    ReadKaldiObject(model_rxfilename, &tmodel);
    
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    CompactLatticeWriter clat_writer(lats_wspecifier); 

    WordBoundaryInfo info(opts, word_boundary_rxfilename);
    
    int32 num_done = 0, num_err = 0; // Note: we may have even in
                                     // error cases.

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      const CompactLattice &clat = clat_reader.Value();

      CompactLattice aligned_clat;
      bool ok = WordAlignLattice(clat, tmodel, info, &aligned_clat);
      
      if (do_test)
        TestWordAlignedLattice(clat, tmodel, info, aligned_clat, ok);

      if (!ok) {
        num_err++;
        if (!output_if_error)
          KALDI_WARN << "Lattice for " << key << " did align correctly";
        else {
          if (aligned_clat.Start() != fst::kNoStateId) {
            KALDI_LOG << "Outputting partial lattice for " << key;
            clat_writer.Write(key, aligned_clat);
          }
        }
      } else {
        if (aligned_clat.Start() == fst::kNoStateId) {
          num_err++;
          KALDI_WARN << "Lattice was empty for key " << key;
        } else {
          num_done++;
          KALDI_VLOG(2) << "Aligned lattice for " << key;
          clat_writer.Write(key, aligned_clat);
        }
      }
    }
    KALDI_LOG << "Successfully aligned " << num_done << " lattices; "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
