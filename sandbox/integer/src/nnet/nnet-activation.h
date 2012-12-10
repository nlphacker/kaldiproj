// nnet/nnet-activation.h

// Copyright 2011  Karel Vesely

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


#ifndef KALDI_NNET_ACTIVATION_H
#define KALDI_NNET_ACTIVATION_H

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out, Nnet *nnet) 
    : Component(dim_in, dim_out, nnet)
  { }
  ~Sigmoid()
  { }

  ComponentType GetType() const {
    return kSigmoid;
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    cu::Sigmoid(in, out);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    cu::DiffSigmoid(out_diff, out, in_diff);
  }
};


class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out, Nnet *nnet) 
    : Component(dim_in, dim_out, nnet)
  { }
  ~Softmax()
  { }

  ComponentType GetType() const {
    return kSoftmax;
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    cu::Softmax(in, out);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }
};



} // namespace

#endif

