// nnet/nnet-parallel-component.h

// Copyright 2014  Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
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


#ifndef KALDI_NNET_NNET_PARALLEL_COMPONENT_H_
#define KALDI_NNET_NNET_PARALLEL_COMPONENT_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class ParallelComponent : public UpdatableComponent {
 public:
  ParallelComponent(int32 dim_in, int32 dim_out) 
    : UpdatableComponent(dim_in, dim_out)
  { }
  ~ParallelComponent()
  { }

  Component* Copy() const { return new ParallelComponent(*this); }
  ComponentType GetType() const { return kParallelComponent; }

  void InitData(std::istream &is) {
    // define options
    std::vector<std::string> nested_nnet_proto;
    std::vector<std::string> nested_nnet_filename;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<NestedNnetFilename>") {
        while(!is.eof()) {
          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</NestedNnetFilename>") break;
          nested_nnet_filename.push_back(file_or_end);
        }
      } else if (token == "<NestedNnetProto>") {
        while(!is.eof()) {
          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</NestedNnetProto>") break;
          nested_nnet_proto.push_back(file_or_end);
        }
      } else KALDI_ERR << "Unknown token " << token << ", typo in config?"
                       << " (NestedNnetFilename|NestedNnetProto)";
      is >> std::ws; // eat-up whitespace
    }
    // initialize
    KALDI_ASSERT((nested_nnet_proto.size() > 0) ^ (nested_nnet_filename.size() > 0)); //xor
    // read nnets from files
    if (nested_nnet_filename.size() > 0) {
      for (int32 i=0; i<nested_nnet_filename.size(); i++) {
        Nnet nnet;
        nnet.Read(nested_nnet_filename[i]);
        nnet_.push_back(nnet);
      }
    }
    // initialize nnets from prototypes
    if (nested_nnet_proto.size() > 0) {
      for (int32 i=0; i<nested_nnet_proto.size(); i++) {
        Nnet nnet;
        nnet.Init(nested_nnet_proto[i]);
        nnet_.push_back(nnet);
      }
    }
    // check dim-sum of nested nnets
    int32 nnet_input_sum = 0, nnet_output_sum = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      nnet_input_sum += nnet_[i].InputDim();
      nnet_output_sum += nnet_[i].OutputDim();
    }
    KALDI_ASSERT(InputDim() == nnet_input_sum);
    KALDI_ASSERT(OutputDim() == nnet_output_sum);
  }

  void ReadData(std::istream &is, bool binary) {
    // read
    ExpectToken(is, binary, "<NestedNnetCount>");
    int32 nnet_count;
    ReadBasicType(is, binary, &nnet_count);
    for (int32 i=0; i<nnet_count; i++) {
      ExpectToken(is, binary, "<NestedNnet>");
      int32 dummy;
      ReadBasicType(is, binary, &dummy);
      Nnet nnet;
      nnet.Read(is, binary);
      nnet_.push_back(nnet);
    }
    ExpectToken(is, binary, "</ParallelComponent>");

    // check dim-sum of nested nnets
    int32 nnet_input_sum = 0, nnet_output_sum = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      nnet_input_sum += nnet_[i].InputDim();
      nnet_output_sum += nnet_[i].OutputDim();
    }
    KALDI_ASSERT(InputDim() == nnet_input_sum);
    KALDI_ASSERT(OutputDim() == nnet_output_sum);
  }

  void WriteData(std::ostream &os, bool binary) const {
    // useful dims
    int32 nnet_count = nnet_.size();
    //
    WriteToken(os, binary, "<NestedNnetCount>");
    WriteBasicType(os, binary, nnet_count);
    for (int32 i=0; i<nnet_count; i++) {
      WriteToken(os, binary, "<NestedNnet>");
      WriteBasicType(os, binary, i+1);
      nnet_[i].Write(os, binary);
    }
    WriteToken(os, binary, "</ParallelComponent>");
  }

  int32 NumParams() const { 
    int32 num_params_sum = 0;
    for (int32 i=0; i<nnet_.size(); i++) 
      num_params_sum += nnet_[i].NumParams();
    return num_params_sum;
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const { KALDI_ERR << "UNIMPLEMENTED"; /*wei_copy->Resize(NumParams()); nnet_.GetParams(wei_copy);*/ }
  std::string Info() const { return ""; /* return std::string("nested_network {\n") + nnet_.Info() + "}\n"; */ }
  std::string InfoGradient() const { return ""; /*return std::string("nested_gradient {\n") + nnet_.InfoGradient() + "}\n";*/ }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    int32 input_offset = 0, output_offset = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      CuSubMatrix<BaseFloat> src(in.ColRange(input_offset, nnet_[i].InputDim()));
      CuSubMatrix<BaseFloat> tgt(out->ColRange(output_offset, nnet_[i].OutputDim()));
      //
      CuMatrix<BaseFloat> tgt_aux;
      nnet_[i].Propagate(CuMatrix<BaseFloat>(src), &tgt_aux); // need CuMatrix<> here
      tgt.CopyFromMat(tgt_aux);
      //
      input_offset += nnet_[i].InputDim();
      output_offset += nnet_[i].OutputDim();
    }
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    int32 input_offset = 0, output_offset = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      CuSubMatrix<BaseFloat> src(out_diff.ColRange(output_offset, nnet_[i].OutputDim()));
      CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(input_offset, nnet_[i].InputDim()));
      //
      CuMatrix<BaseFloat> tgt_aux;
      nnet_[i].Backpropagate(CuMatrix<BaseFloat>(src), &tgt_aux); // need CuMatrix<> here
      tgt.CopyFromMat(tgt_aux);
      //
      input_offset += nnet_[i].InputDim();
      output_offset += nnet_[i].OutputDim();
    }
  }

  void Update(const CuMatrix<BaseFloat> &input, const CuMatrix<BaseFloat> &diff) {
    ; // do nothing
  }
 
  void SetTrainOptions(const NnetTrainOptions &opts) {
    for (int32 i=0; i<nnet_.size(); i++) {
      nnet_[i].SetTrainOptions(opts);
    }
  }

 private:
  std::vector<Nnet> nnet_;
};

} // namespace nnet1
} // namespace kaldi

#endif
