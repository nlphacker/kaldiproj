// tree/clusterable-classes.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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

#include <algorithm>
#include <string>
#include "base/kaldi-math.h"
#include "itf/clusterable-itf.h"
#include "tree/clusterable-classes.h"

namespace kaldi {

// ============================================================================
// Implementations common to all Clusterable classes (may be overridden for
// speed).
// ============================================================================

BaseFloat Clusterable::ObjfPlus(const Clusterable &other) const {
  Clusterable *copy = this->Copy();
  copy->Add(other);
  BaseFloat ans = copy->Objf();
  delete copy;
  return ans;
}

BaseFloat Clusterable::ObjfMinus(const Clusterable &other) const {
  Clusterable *copy = this->Copy();
  copy->Sub(other);
  BaseFloat ans = copy->Objf();
  delete copy;
  return ans;
}

BaseFloat Clusterable::Distance(const Clusterable &other) const {
  Clusterable *copy = this->Copy();
  copy->Add(other);
  BaseFloat ans = this->Objf() + other.Objf() - copy->Objf();
  if (ans < 0) {
    // This should not happen. Check if it is more than just rounding error.
    if (std::fabs(ans) > 0.01 * (1.0 + std::fabs(copy->Objf()))) {
      KALDI_WARN << "Negative number returned (badly defined Clusterable "
                 << "class?): ans= " << ans;
    }
    ans = 0;
  }
  delete copy;
  return ans;
}

// ============================================================================
// Implementation of ScalarClusterable class.
// ============================================================================

BaseFloat ScalarClusterable::Objf() const {
  if (count_ == 0) {
    return 0;
  } else {
    assert(count_ > 0);
    return -(x2_ - x_ * x_ / count_);
  }
}

void ScalarClusterable::Add(const Clusterable &other_in) {
  assert(other_in.Type() == "scalar");
  const ScalarClusterable *other =
      static_cast<const ScalarClusterable*>(&other_in);
  x_ += other->x_;
  x2_ += other->x2_;
  count_ += other->count_;
}

void ScalarClusterable::Sub(const Clusterable &other_in) {
  assert(other_in.Type() == "scalar");
  const ScalarClusterable *other =
      static_cast<const ScalarClusterable*>(&other_in);
  x_ -= other->x_;
  x2_ -= other->x2_;
  count_ -= other->count_;
}

Clusterable* ScalarClusterable::Copy() const {
  ScalarClusterable *ans = new ScalarClusterable();
  ans->Add(*this);
  return ans;
}

void ScalarClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "SCL");  // magic string.
  WriteBasicType(os, binary, x_);
  WriteBasicType(os, binary, x2_);
  WriteBasicType(os, binary, count_);
}

Clusterable* ScalarClusterable::Read(std::istream &is, bool binary) const {
  ScalarClusterable *sc = new ScalarClusterable();
  sc->Read_(is, binary);
  return sc;
}

void ScalarClusterable::Read_(std::istream &is, bool binary) {
  ExpectToken(is, binary, "SCL");
  ReadBasicType(is, binary, &x_);
  ReadBasicType(is, binary, &x2_);
  ReadBasicType(is, binary, &count_);
}

std::string ScalarClusterable::Info() {
  std::stringstream str;
  if (count_ == 0) {
    str << "[empty]";
  } else {
    str << "[mean " << (x_ / count_) << ", var " << (x2_ / count_ -
        (x_ * x_ / (count_ * count_))) << "]";
  }
  return str.str();
}

// ============================================================================
// Implementation of GaussClusterable class.
// ============================================================================

void GaussClusterable::AddStats(const VectorBase<BaseFloat> &vec,
                                BaseFloat weight) {
  count_ += weight;
  stats_.Row(0).AddVec(weight, vec);
  stats_.Row(1).AddVec2(weight, vec);
}

void GaussClusterable::Add(const Clusterable &other_in) {
  assert(other_in.Type() == "gauss");
  const GaussClusterable *other =
      static_cast<const GaussClusterable*>(&other_in);
  count_ += other->count_;
  stats_.AddMat(1.0, other->stats_);
}

void GaussClusterable::Sub(const Clusterable &other_in) {
  assert(other_in.Type() == "gauss");
  const GaussClusterable *other =
      static_cast<const GaussClusterable*>(&other_in);
  count_ -= other->count_;
  stats_.AddMat(-1.0, other->stats_);
}

Clusterable* GaussClusterable::Copy() const {
  assert(stats_.NumRows() == 2);
  GaussClusterable *ans = new GaussClusterable(stats_.NumCols(), var_floor_);
  ans->Add(*this);
  return ans;
}

void GaussClusterable::Scale(BaseFloat f) {
  assert(f >= 0.0);
  count_ *= f;
  stats_.Scale(f);
}

void GaussClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "GCL");  // magic string.
  WriteBasicType(os, binary, count_);
  WriteBasicType(os, binary, var_floor_);
  stats_.Write(os, binary);
}

Clusterable* GaussClusterable::Read(std::istream &is, bool binary) const {
  GaussClusterable *gc = new GaussClusterable();
  gc->Read_(is, binary);
  return gc;
}

void GaussClusterable::Read_(std::istream &is, bool binary) {
  ExpectToken(is, binary, "GCL");  // magic string.
  ReadBasicType(is, binary, &count_);
  ReadBasicType(is, binary, &var_floor_);
  stats_.Read(is, binary);
}

BaseFloat GaussClusterable::Objf() const {
  if (count_ <= 0.0) {
    if (count_ < -0.1) {
      KALDI_WARN << "GaussClusterable::Objf(), count is negative " << count_;
    }
    return 0.0;
  } else {
    size_t dim = stats_.NumCols();
    Vector<double> vars(dim);
    double objf_per_frame = 0.0;
    for (size_t d = 0; d < dim; d++) {
      double mean(stats_(0, d) / count_), var = stats_(1, d) / count_ - mean
          * mean, floored_var = std::max(var, var_floor_);
      vars(d) = floored_var;
      objf_per_frame += -0.5 * var / floored_var;
    }
    objf_per_frame += -0.5 * (vars.SumLog() + M_LOG_2PI * dim);
    if (KALDI_ISNAN(objf_per_frame)) {
      KALDI_WARN << "GaussClusterable::Objf(), objf is NaN\n";
      return 0.0;
    }
    // KALDI_VLOG(2) << "count = " << count_ << ", objf_per_frame = "<< objf_per_frame
    //   << ", returning " << (objf_per_frame*count_) << ", floor = " << var_floor_;
    return objf_per_frame * count_;
  }
}


}  // end namespace kaldi.
