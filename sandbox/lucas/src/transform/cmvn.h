// transform/cmvn.h

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


#ifndef KALDI_TRANSFORM_CMVN_H_
#define KALDI_TRANSFORM_CMVN_H_

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "hmm/transition-model.h"

namespace kaldi {

/// This function initializes the matrix to dimension 2 by (dim+1);
/// 1st "dim" elements of 1st row are mean stats, 1st "dim" elements
/// of 2nd row are var stats, last element of 1st row is count,
/// last element of 2nd row is zero.
void InitCmvnStats(int32 dim, Matrix<double> *stats);

/// Accumulation from a single frame (weighted).
void AccCmvnStats(const VectorBase<BaseFloat> &feats, BaseFloat weight, MatrixBase<double> *stats);

/// Accumulation from a feature file (possibly weighted-- useful in excluding silence).
void AccCmvnStats(const MatrixBase<BaseFloat> &feats,
                  const VectorBase<BaseFloat> *weights,  // or NULL
                  MatrixBase<double> *stats);



/// Apply cepstral mean and variance normalization to a matrix of features.
void ApplyCmvn(const MatrixBase<double> &stats,
               bool norm_vars,
               MatrixBase<BaseFloat> *feats);


}  // namespace kaldi

#endif  // KALDI_TRANSFORM_CMVN_H_