// transform/lda-estimate.h

// Copyright 2009-2011  Jan Silovsky

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

#ifndef KALDI_TRANSFORM_LDA_ESTIMATE_H_
#define KALDI_TRANSFORM_LDA_ESTIMATE_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

struct LdaEstimateOptions {
  bool remove_offset;
  int32 dim;
  bool allow_large_dim;
  LdaEstimateOptions(): remove_offset(false), dim(40), allow_large_dim(false) { }

  void Register(ParseOptions *po) {
    po->Register("remove-offset", &remove_offset, "If true, output an affine "
                 "transform that makes the projected data mean equal to zero.");
    po->Register("dim", &dim, "Dimension to project to with LDA");
    po->Register("allow-large-dim", &allow_large_dim, "If true, allow an LDA "
                 "dimension larger than the number of classes.");
  }    
};

/** Class for computing linear discriminant analysis (LDA) transform.
    C.f. \ref transform_lda.
 */
class LdaEstimate {
 public:
  LdaEstimate() {}

  /// Allocates memory for accumulators
  void Init(int32 num_classes, int32 dimension);
  /// Returns the number of classes
  int32 NumClasses() const { return first_acc_.NumRows(); }
  /// Returns the dimensionality of the feature vectors
  int32 Dim() const { return first_acc_.NumCols(); }
  /// Sets all accumulators to zero
  void ZeroAccumulators();
  /// Scales all accumulators
  void Scale(BaseFloat f);

  /// Accumulates data
  void Accumulate(const VectorBase<BaseFloat> &data, int32 class_id, BaseFloat weight = 1.0);

  /// Estimates the LDA transform matrix m.  If Mfull != NULL, it also outputs
  /// the full matrix (without dimensionality reduction), which is useful for
  /// some purposes.  If remove_offset == true, it will output both matrices
  /// with an extra column which corresponds to mean-offset removal (the matrix
  /// should be multiplied by the feature with a 1 appended to give the correct
  /// result, as with other Kaldi transforms.)
  /// The "remove_offset" argument is new and should be set to false for back
  /// compatibility.
  void Estimate(const LdaEstimateOptions &opts, 
                Matrix<BaseFloat> *M,
                Matrix<BaseFloat> *Mfull = NULL) const;

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

 private:
  Vector<double> zero_acc_;
  Matrix<double> first_acc_;
  SpMatrix<double> total_second_acc_;

  /// This function modifies the LDA matrix so that it
  /// also subtracts the mean feature value.
  void AddMeanOffset(const VectorBase<double> &total_mean,
                     Matrix<BaseFloat> *projection) const;
  
  // Disallow assignment operator.
  LdaEstimate &operator = (const LdaEstimate &other);
};

}  // End namespace kaldi

#endif  // KALDI_TRANSFORM_LDA_ESTIMATE_H_

