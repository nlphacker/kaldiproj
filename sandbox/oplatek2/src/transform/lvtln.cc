// transform/lvtln.cc

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

#include <utility>
#include <vector>
using std::vector;

#include "transform/lvtln.h"

namespace kaldi {

LinearVtln::LinearVtln(int32 dim, int32 num_classes, int32 default_class) {
  default_class_ = default_class;
  KALDI_ASSERT(default_class >= 0 && default_class < num_classes);
  A_.resize(num_classes);
  for (int32 i = 0; i < num_classes; i++) {
    A_[i].Resize(dim, dim);
    A_[i].SetUnit();
  }
  logdets_.clear();
  logdets_.resize(num_classes, 0.0);
} // namespace kaldi


BaseFloat LinearVtln::GetDefaultAuxf(const FmllrDiagGmmAccs &speaker_stats) const {
  Matrix<BaseFloat> default_mat(Dim(), Dim()+1);
  default_mat.SetUnit();
  return FmllrAuxFuncDiagGmm(default_mat, speaker_stats);
}

BaseFloat LinearVtln::GetAuxf(const FmllrDiagGmmAccs &speaker_stats,
                              BaseFloat logdet_scale,
                              int32 class_idx,
                              const VectorBase<BaseFloat> &offset) const {
  assert(class_idx >= 0 && class_idx < NumClasses());
  int32 dim = Dim();

  Matrix<BaseFloat> mat(dim, dim+1);
  {  // construct the fMLLR matrix "mat"
    SubMatrix<BaseFloat> square(mat, 0, dim, 0, dim);
    square.CopyFromMat(A_[class_idx]);
    for (int32 i = 0; i < dim; i++) mat(i, dim) = offset(i);
  }
  BaseFloat extra_logdet = 0.0;
  if (logdet_scale != 1.0) extra_logdet = logdets_[class_idx]*(logdet_scale-1.0);
  return FmllrAuxFuncDiagGmm(mat, speaker_stats)
      + speaker_stats.beta_*extra_logdet;
}

void LinearVtln::Read(std::istream &is, bool binary) {
  int32 sz;
  ExpectToken(is, binary, "<LinearVtln>");
  ReadBasicType(is, binary, &sz);
  A_.resize(sz);
  logdets_.resize(sz);
  for (int32 i = 0; i < sz; i++) {
    A_[i].Read(is, binary);
    ReadBasicType(is, binary, &(logdets_[i]));
  }
  ExpectToken(is, binary, "</LinearVtln>");
}

void LinearVtln::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<LinearVtln>");
  if(!binary) os << "\n";
  int32 sz = A_.size();
  assert(static_cast<size_t>(sz) == logdets_.size());
  WriteBasicType(os, binary, sz);
  for (int32 i = 0; i < sz; i++) {
    A_[i].Write(os, binary);
    WriteBasicType(os, binary, logdets_[i]);
    if(!binary) os << "\n";
  }
  WriteToken(os, binary, "</LinearVtln>");
}


/// Compute the transform for the speaker.
void LinearVtln::ComputeTransform(const FmllrDiagGmmAccs &accs,
                                  std::string norm_type,  // "none", "offset", "diag"
                                  MatrixBase<BaseFloat> *Ws,  // output fMLLR transform, should be size dim x dim+1
                                  int32 *class_idx,  // the transform that was chosen...
                                  BaseFloat *logdet_out,
                                  BaseFloat *objf_impr,  // versus no transform
                                  BaseFloat *count) {
  int32 dim = Dim();
  KALDI_ASSERT(dim != 0);
  if (norm_type != "none"  && norm_type != "offset" && norm_type != "diag")
    KALDI_ERR << "LinearVtln::ComputeTransform, norm_type should be "
        "one of \"none\", \"offset\" or \"diag\"";

  if (accs.beta_ == 0.0) {
    KALDI_WARN << "no stats, returning default transform";
    *class_idx = default_class_;
    int32 dim = Dim();
    KALDI_ASSERT(Ws != NULL && Ws->NumRows() == dim && Ws->NumCols() == dim+1);
    Ws->Range(0, dim, 0, dim).CopyFromMat(A_[default_class_]);
    Ws->Range(0, dim, dim, 1).SetZero();  // Set last column to zero.
    if (logdet_out) *logdet_out = logdets_[default_class_];
    if (objf_impr) *objf_impr = 0;
    if (count) *count = 0;
  }

  Matrix<BaseFloat> best_transform(dim, dim+1);
  best_transform.SetUnit();
  BaseFloat old_objf = FmllrAuxFuncDiagGmm(best_transform, accs),
      best_objf = -1.0e+100;
  int32 best_class = -1;

  for (int32 i = 0; i < NumClasses(); i++) {
    FmllrDiagGmmAccs accs_tmp(accs);
    ApplyFeatureTransformToStats(A_[i], &accs_tmp);
    // "old_trans" just needed by next function as "initial" transform.
    Matrix<BaseFloat> old_trans(dim, dim+1); old_trans.SetUnit();
    Matrix<BaseFloat> trans(dim, dim+1);
    ComputeFmllrMatrixDiagGmm(old_trans, accs_tmp, norm_type,
                              100,  // num iters.. don't care since norm_type != "full"
                              &trans);
    Matrix<BaseFloat> product(dim, dim+1);
    // product = trans * A_[i] (modulo messing about with offsets)
    ComposeTransforms(trans, A_[i], false, &product);

    BaseFloat objf = FmllrAuxFuncDiagGmm(product, accs);
    if (objf > best_objf) {
      best_objf = objf;
      best_class = i;
      best_transform.CopyFromMat(product);
    }
  }
  KALDI_ASSERT(best_class != -1);
  if (Ws) Ws->CopyFromMat(best_transform);
  if (class_idx) *class_idx = best_class;
  if (logdet_out) *logdet_out = logdets_[best_class];
  if (objf_impr) *objf_impr = best_objf - old_objf;
  if (count) *count = accs.beta_;
}



void LinearVtln::SetTransform(int32 i, const MatrixBase<BaseFloat> &transform) {
  KALDI_ASSERT(i >= 0 && i < NumClasses());
  KALDI_ASSERT(transform.NumRows() == transform.NumCols() &&
               static_cast<int32>(transform.NumRows()) == Dim());
  A_[i].CopyFromMat(transform);
  logdets_[i] = A_[i].LogDet();
}


void LinearVtln::GetTransform(int32 i, MatrixBase<BaseFloat> *transform) const {
  KALDI_ASSERT(i >= 0 && i < NumClasses());
  KALDI_ASSERT(transform->NumRows() == transform->NumCols() &&
               static_cast<int32>(transform->NumRows()) == Dim());
  transform->CopyFromMat(A_[i]);
}



} // end namespace kaldi

