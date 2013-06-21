// nnet-cpubin/nnet-subset-egs.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Creates a random subset of the input examples, of a specified size.\n"
        "Uses no more memory than the size of the subset.\n"
        "\n"
        "Usage:  nnet-subset-egs [options] <egs-rspecifier> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet-get-egs [args] ark:- | nnet-subset-egs --n=1000 ark:- ark:subset.egs\n";
    
    int32 srand_seed = 0;
    int32 n = 1000;
    bool randomize_order = true;
    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("n", &n, "Number of examples to output");
    po.Register("randomize-order", &randomize_order, "If true, randomize the order "
                "of the output");
    
    po.Read(argc, argv);
    
    srand(srand_seed);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    std::vector<NnetTrainingExample> egs;
    
    SequentialNnetTrainingExampleReader example_reader(examples_rspecifier);

    int64 num_read = 0;
    for (; !example_reader.Done(); example_reader.Next()) {
      num_read++;
      if (num_read <= n)
        egs.push_back(example_reader.Value());
      else {
        BaseFloat keep_prob = n / static_cast<BaseFloat>(num_read);
        if (WithProb(keep_prob)) { // With probability "keep_prob"
          egs[RandInt(0, n-1)] = example_reader.Value();
        }
      }
    }
    if (randomize_order)
      std::random_shuffle(egs.begin(), egs.end());

    NnetTrainingExampleWriter writer(examples_wspecifier);
    for (size_t i = 0; i < egs.size(); i++) {
      std::ostringstream key;
      key << i;
      writer.Write(key.str(), egs[i]);
    }
    
    KALDI_LOG << "Selected a subset of " << egs.size() << " out of " << num_read
              << " neural-network training examples ";
    
    return (num_read != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


