// gmmbin/gmm-sum-accs.cc
// Copyright 2009-2011  Arnab Ghoshal Microsoft Corporation

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

#include "util/common-utils.h"
#include "gmm/estimate-am-diag-gmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for GMM training.\n"
        "Usage: sum-gmm-accs [options] stats-out stats-in1 stats-in2 ...\n";

    bool binary = false;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_out_filename = po.GetArg(1);
    kaldi::Vector<double> transition_accs;
    kaldi::MlEstimateAmDiagGmm gmm_accs;

    for (int i = 2, max = po.NumArgs(); i <= max; ++i) {
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input is(stats_in_filename, &binary_read);
      transition_accs.Read(is.Stream(), binary_read, true /*add read values*/);
      gmm_accs.Read(is.Stream(), binary_read, true /*add read values*/);
    }

    // Write out the accs
    {
      kaldi::Output os(stats_out_filename, binary);
      transition_accs.Write(os.Stream(), binary);
      gmm_accs.Write(os.Stream(), binary);
    }

    KALDI_LOG << "Written stats to " << stats_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


