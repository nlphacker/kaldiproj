// gmmbin/transf-to-nnet.cc

// Copyright 2012  Brno University of Technology

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
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-various.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert transformation matrix to <biasedlinearity>\n"
        "Usage:  transf-to-nnet [options] <transf-in> <nnet-out>\n"
        "e.g.:\n"
        " transf-to-nnet --binary=false transf.mat nnet.mdl\n";


    bool binary_write = false;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_rxfilename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    //read the matrix
    Matrix<double> cmvn_stats;
    {
      bool binary_read;
      Input ki(transform_rxfilename, &binary_read);
      cmvn_stats.Read(ki.Stream(), binary_read);
    }
    KALDI_ASSERT(cmvn_stats.NumRows() == 2);
    KALDI_ASSERT(cmvn_stats.NumCols() > 1);

    //get the count
    double count = cmvn_stats(0,cmvn_stats.NumCols()-1);
   
    //buffers for shift and scale 
    Vector<BaseFloat> shift(cmvn_stats.NumCols()-1);
    Vector<BaseFloat> scale(cmvn_stats.NumCols()-1);
    
    //compute the shift and scale per each dimension
    for(int32 d=0; d<cmvn_stats.NumCols()-1; d++) {
      BaseFloat mean = cmvn_stats(0,d)/count;
      BaseFloat var = cmvn_stats(1,d)/count - mean*mean;
      scale(d) = 1.0 / sqrt(var);
      shift(d) = -mean * scale(d);
    }

    //we will put the shift and scale to the nnet
    Nnet nnet;

    //create the shift component
    {
      AddShift* shift_component = new AddShift(shift.Dim(),shift.Dim(),&nnet);
      //the pointer will be given to the nnet, so we don't need to call delete
      
      //convert Vector to CuVector
      CuVector<BaseFloat> cu_shift;
      cu_shift.CopyFromVec(shift);

      //set the weights
      shift_component->SetShiftVec(cu_shift);

      //append layer to the nnet
      nnet.AppendLayer(shift_component);
    }

    //create the scale component
    {
      Rescale* scale_component = new Rescale(scale.Dim(),scale.Dim(),&nnet);
      //the pointer will be given to the nnet, so we don't need to call delete
      
      //convert Vector to CuVector
      CuVector<BaseFloat> cu_scale;
      cu_scale.CopyFromVec(scale);

      //set the weights
      scale_component->SetScaleVec(cu_scale);

      //append layer to the nnet
      nnet.AppendLayer(scale_component);
    }

      
    //write the nnet
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


