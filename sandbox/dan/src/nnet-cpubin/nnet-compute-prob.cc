// nnet-cpubin/nnet-compute-prob.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "cudamatrix/cu-common.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints the average log-prob per frame of the given data with a\n"
        "neural net.  The input of this is the output of e.g. nnet-get-egs\n"
        "Aside from the logging output, which goes to the standard error, this program\n"
        "prints the average log-prob per frame to the standard output.\n"
        "Also see nnet-logprob, which produces a matrix of log-probs for each utterance.\n"
        "\n"
        "Usage:  nnet-compute-prob [options] <model-in> <training-examples-in>\n"
        "\n"
        "nnet-randomize-frames [args] | nnet-compute-prob 1.nnet ark:-\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2);

    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }


    std::vector<NnetTrainingExample> examples;
    double tot_like = 0, tot_frames = 0;
    int64 num_examples = 0;
    SequentialNnetTrainingExampleReader example_reader(examples_rspecifier);
    for (; !example_reader.Done(); example_reader.Next(), num_examples++) {
      if (examples.size() == 1000) {
        tot_like += ComputeNnetObjf(am_nnet.GetNnet(), examples);
        tot_frames += examples.size();
        examples.clear();
      }
      examples.push_back(example_reader.Value());
      if (num_examples % 5000 == 0 && num_examples > 0)
        KALDI_LOG << "Saw " << num_examples << " examples, average "
                  << "probability is " << (tot_like / tot_frames) << " with "
                  << "total weight " << tot_frames;
    }
    if (!examples.empty()) {
      tot_like += ComputeNnetObjf(am_nnet.GetNnet(), examples);
      tot_frames += examples.size();
    }

    KALDI_LOG << "Saw " << num_examples << " examples, average "
              << "probability is " << (tot_like / tot_frames) << " with "
              << "total weight " << tot_frames;
    
    std::cout << (tot_like / tot_frames) << "\n";
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


