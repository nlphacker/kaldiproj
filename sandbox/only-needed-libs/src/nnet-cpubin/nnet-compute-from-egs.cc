// nnet-cpubin/nnet-compute-from-egs.cc

// Copyright 2012-2013  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet-cpu/train-nnet.h"
#include "nnet-cpu/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Does the neural net computation, taking as input the nnet-training examples\n"
        "(typically an archive with the extension .egs), ignoring the labels; it\n"
        "outputs as a matrix the result.  Used mostly for debugging.\n"
        "\n"
        "Usage:  nnet-compute-from-egs [options] <raw-nnet-in> <egs-rspecifier> "
        "<feature-wspecifier>\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string raw_nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        features_or_loglikes_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);
    
    int64 num_egs = 0;

    SequentialNnetTrainingExampleReader example_reader(examples_rspecifier);
    BaseFloatMatrixWriter writer(features_or_loglikes_wspecifier);
    
    int32 left_context = nnet.LeftContext(),
        context = nnet.LeftContext() + 1 + nnet.RightContext();

    for (; !example_reader.Done(); example_reader.Next()) {
      const NnetTrainingExample &eg = example_reader.Value();
      int32 start_dim = eg.left_context - left_context;
      SubMatrix<BaseFloat> input_block(eg.input_frames,
                                       start_dim, context,
                                       0, eg.input_frames.NumCols());
      Matrix<BaseFloat> output_block(1, nnet.OutputDim());
      bool pad_input = false;
      NnetComputation(nnet, input_block, eg.spk_info, pad_input, &output_block);
      writer.Write("global", output_block);
      num_egs++;
    }
    
    KALDI_LOG << "Processed " << num_egs << " examples.";
    
    return (num_egs == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


