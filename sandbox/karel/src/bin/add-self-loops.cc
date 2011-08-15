// bin/add-self-loops.cc
// Copyright 2009-2011 Microsoft Corporation

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

#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"

/** @brief Add self-loops and transition probabilities to transducer, expanding to transition-ids.
*/
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Add self-loops and transition probabilities to transducer, expanding to transition-ids\n"
        "Usage:   add-self-loops [options] transition-gmm/acoustic-model [fst-in] [fst-out]\n"
        "e.g.: \n"
        " add-self-loops --self-loop-scale = 0.1 1.mdl < HCLG_noloops.fst > HCLG_full.fst\n";

    BaseFloat self_loop_scale = 1.0;
    bool reorder = true;
    std::string disambig_in_filename;

    ParseOptions po(usage);
    po.Register("self-loop-scale", &self_loop_scale, "Scale for self-loop probabilities relative to LM.");
    po.Register("disambig-syms", &disambig_in_filename, "List of disambiguation symbols on input of fst-in [input file]");
    po.Register("reorder", &reorder, "If true, reorder symbols for more decoding efficiency");
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1);
    std::string fst_in_filename = po.GetOptArg(2);
    if (fst_in_filename == "-") fst_in_filename = "";
    std::string fst_out_filename = po.GetOptArg(3);
    if (fst_out_filename == "-") fst_out_filename = "";


    std::vector<int32> disambig_syms_in;
    if (disambig_in_filename != "") {
      if (disambig_in_filename == "-") disambig_in_filename = "";
      if (!ReadIntegerVectorSimple(disambig_in_filename, &disambig_syms_in))
        KALDI_EXIT << "add-self-loops: could not read disambig symbols from "
                   <<(disambig_in_filename == "" ?
                      "standard input" : disambig_in_filename);
    }

    TransitionModel trans_model;
    {
      bool binary_in;
      Input ki(model_in_filename, &binary_in);
      trans_model.Read(ki.Stream(), binary_in);
    }


    fst::VectorFst<fst::StdArc> *fst =
        fst::VectorFst<fst::StdArc>::Read(fst_in_filename);
    if (!fst)
      KALDI_EXIT << "add-self-loops: error reading input FST.";


    // The work gets done here.
    AddSelfLoops(trans_model,
                 disambig_syms_in,
                 self_loop_scale,
                 reorder,
                 fst);

    if (! fst->Write(fst_out_filename) )
      KALDI_EXIT << "add-self-loops: error writing FST to "
                 << (fst_out_filename == "" ?
                     "standard output" : fst_out_filename);

    delete fst;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

