// bin/copy-matrix.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "matrix/kaldi-matrix.h"
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy matrices, or archives of matrices (e.g. features or transforms)\n"
        "\n"
        "Usage: copy-matrix [options] (matrix-in-rspecifier|matrix-in-rxfilename) (matrix-out-wspecifier|matrix-out-wxfilename)\n"
        " e.g.: copy-matrix --binary=false 1.mat -\n"
        "   copy-matrix ark:2.trans ark,t:-\n";
    
    bool binary = true;
    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string matrix_in_fn = po.GetArg(1),
        matrix_out_fn = po.GetArg(2);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(matrix_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_rspecifier =
        (ClassifyRspecifier(matrix_out_fn, NULL, NULL)
         != kNoRspecifier);

    if (in_is_rspecifier != out_is_rspecifier)
      KALDI_ERR << "Cannot mix archives with regular files (copying matrices)\n";
    
    if (!in_is_rspecifier) {
      Matrix<BaseFloat> mat;
      {
        bool binary_in;
        Input ki(matrix_in_fn, &binary_in);
        mat.Read(ki.Stream(), binary_in);
      }
      Output ko(matrix_out_fn, binary);
      mat.Write(ko.Stream(), binary);
      KALDI_LOG << "Copied matrix to " << matrix_out_fn;
      return 0;
    } else {
      int num_done = 0;
      BaseFloatMatrixWriter writer(matrix_out_fn);
      SequentialBaseFloatMatrixReader reader(matrix_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++)
        writer.Write(reader.Key(), reader.Value());
      KALDI_LOG << "Copied " << num_done << " matrices.";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


