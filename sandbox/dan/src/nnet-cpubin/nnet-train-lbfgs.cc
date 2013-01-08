// nnet-cpubin/nnet-train-lbfgs.cc

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
#include "nnet-cpu/nnet-lbfgs.h"
#include "nnet-cpu/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train the neural network parameters using L-BFGS on a subset of data\n"
        "\n"
        "Usage:  nnet-train-lbfgs [options] <model-in> <training-examples-in> <model-out>\n"
        "\n"
        "e.g.:\n"
        "nnet-randomize-frames [args] | nnet-train-lbfgs 1.nnet ark:- 2.nnet\n";
    
    bool binary_write = true;
    bool zero_stats = true;
    NnetLbfgsTrainerConfig train_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("zero-stats", &zero_stats, "If true, zero occupation "
                "counts stored with the neural net (only affects mixing up).");

    train_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);


    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    if (zero_stats) am_nnet.GetNnet().ZeroStats();

    NnetLbfgsTrainer trainer(train_config);
    
    int64 num_examples = 0;
      
    SequentialNnetTrainingExampleReader example_reader(examples_rspecifier);
    for (; !example_reader.Done(); example_reader.Next(), num_examples++)
      trainer.AddTrainingExample(example_reader.Value());

    trainer.Train(&(am_nnet.GetNnet()));
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.  Wrote model to "
              << nnet_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


