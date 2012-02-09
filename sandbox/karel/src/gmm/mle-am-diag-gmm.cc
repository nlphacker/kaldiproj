// gmm/mle-am-diag-gmm.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation;
//                      Georg Stemmer; Yanmin Qian

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

#include "gmm/am-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

const AccumDiagGmm& AccumAmDiagGmm::GetAcc(int32 index) const {
  assert(index >= 0 && index < static_cast<int32>(gmm_accumulators_.size()));
  return *(gmm_accumulators_[index]);
}

AccumDiagGmm& AccumAmDiagGmm::GetAcc(int32 index) {
  assert(index >= 0 && index < static_cast<int32>(gmm_accumulators_.size()));
  return *(gmm_accumulators_[index]);
}

AccumAmDiagGmm::~AccumAmDiagGmm() {
  DeletePointers(&gmm_accumulators_);
}

void AccumAmDiagGmm::Init(const AmDiagGmm &model,
                              GmmFlagsType flags) {
  DeletePointers(&gmm_accumulators_);  // in case was non-empty when called.
  gmm_accumulators_.resize(model.NumPdfs(), NULL);
  for (int32 i = 0; i < model.NumPdfs(); i++) {
    gmm_accumulators_[i] = new AccumDiagGmm();
    gmm_accumulators_[i]->Resize(model.GetPdf(i), flags);
  }
}

void AccumAmDiagGmm::Init(const AmDiagGmm &model,
                              int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(dim > 0);
  DeletePointers(&gmm_accumulators_);  // in case was non-empty when called.
  gmm_accumulators_.resize(model.NumPdfs(), NULL);
  for (int32 i = 0; i < model.NumPdfs(); i++) {
    gmm_accumulators_[i] = new AccumDiagGmm();
    gmm_accumulators_[i]->Resize(model.GetPdf(i).NumGauss(),
                                 dim, flags);
  }
}

void AccumAmDiagGmm::SetZero(GmmFlagsType flags) {
  for (size_t i = 0; i < gmm_accumulators_.size(); ++i) {
    gmm_accumulators_[i]->SetZero(flags);
  }
}

BaseFloat AccumAmDiagGmm::AccumulateForGmm(
    const AmDiagGmm &model, const VectorBase<BaseFloat>& data,
    int32 gmm_index, BaseFloat weight) {
  KALDI_ASSERT(static_cast<size_t>(gmm_index) < gmm_accumulators_.size());
  return gmm_accumulators_[gmm_index]->AccumulateFromDiag(model.GetPdf(
      gmm_index), data, weight);
}

BaseFloat AccumAmDiagGmm::AccumulateForGmmTwofeats(
    const AmDiagGmm &model,
    const VectorBase<BaseFloat>& data1,
    const VectorBase<BaseFloat>& data2,
    int32 gmm_index,
    BaseFloat weight) {
  assert(static_cast<size_t>(gmm_index) < gmm_accumulators_.size());
  const DiagGmm &gmm = model.GetPdf(gmm_index);
  AccumDiagGmm &acc = *(gmm_accumulators_[gmm_index]);
  Vector<BaseFloat> posteriors;
  BaseFloat log_like = gmm.ComponentPosteriors(data1, &posteriors);
  posteriors.Scale(weight);
  acc.AccumulateFromPosteriors(data2, posteriors);
  return log_like;
}


void AccumAmDiagGmm::AccumulateFromPosteriors(
    const AmDiagGmm &model, const VectorBase<BaseFloat>& data,
    int32 gmm_index, const VectorBase<BaseFloat>& posteriors) {
  KALDI_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());
  gmm_accumulators_[gmm_index]->AccumulateFromPosteriors(data, posteriors);
}

void AccumAmDiagGmm::AccumulateForGaussian(
    const AmDiagGmm &am, const VectorBase<BaseFloat>& data,
    int32 gmm_index, int32 gauss_index, BaseFloat weight) {
  KALDI_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());
  KALDI_ASSERT(gauss_index >= 0
      && gauss_index < am.GetPdf(gmm_index).NumGauss());
  gmm_accumulators_[gmm_index]->AccumulateForComponent(data, gauss_index, weight);
}

void AccumAmDiagGmm::Read(std::istream& in_stream, bool binary,
                               bool add) {
  int32 num_pdfs;
  ExpectMarker(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && gmm_accumulators_.empty())) {
    gmm_accumulators_.resize(num_pdfs, NULL);
    for (std::vector<AccumDiagGmm*>::iterator it = gmm_accumulators_.begin(),
             end = gmm_accumulators_.end(); it != end; ++it) {
      if (*it != NULL) delete *it;
      *it = new AccumDiagGmm();
      (*it)->Read(in_stream, binary, add);
    }
  } else {
    if (gmm_accumulators_.size() != static_cast<size_t> (num_pdfs))
      KALDI_ERR << "Adding accumulators but num-pdfs do not match: "
                << (gmm_accumulators_.size()) << " vs. "
                << (num_pdfs);
    for (std::vector<AccumDiagGmm*>::iterator it = gmm_accumulators_.begin(),
             end = gmm_accumulators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);
  }
}

void AccumAmDiagGmm::Write(std::ostream& out_stream, bool binary) const {
  int32 num_pdfs = gmm_accumulators_.size();
  WriteMarker(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs);
  for (std::vector<AccumDiagGmm*>::const_iterator it =
      gmm_accumulators_.begin(), end = gmm_accumulators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}


BaseFloat AccumAmDiagGmm::TotCount() const {
  BaseFloat ans = 0.0;
  for (int32 pdf = 0; pdf < NumAccs(); pdf++)
    ans += gmm_accumulators_[pdf]->occupancy().Sum();
  return ans;
}

void MleAmDiagGmmUpdate(const MleDiagGmmOptions &config,
            const AccumAmDiagGmm &amdiaggmm_acc,
            GmmFlagsType flags,
            AmDiagGmm *am_gmm,
            BaseFloat *obj_change_out,
            BaseFloat *count_out) {
  KALDI_ASSERT(am_gmm != NULL);
  KALDI_ASSERT(amdiaggmm_acc.NumAccs() == am_gmm->NumPdfs());
  if (obj_change_out != NULL) *obj_change_out = 0.0;
  if (count_out != NULL) *count_out = 0.0;
  BaseFloat tmp_obj_change, tmp_count;
  BaseFloat *p_obj = (obj_change_out != NULL) ? &tmp_obj_change : NULL,
            *p_count   = (count_out != NULL) ? &tmp_count : NULL;

  for (size_t i = 0; i < amdiaggmm_acc.NumAccs(); i++) {
    MleDiagGmmUpdate(config, amdiaggmm_acc.GetAcc(i), flags, &(am_gmm->GetPdf(i)), p_obj,
        p_count);

    if (obj_change_out != NULL) *obj_change_out += tmp_obj_change;
    if (count_out != NULL) *count_out += tmp_count;
  }
}

}  // namespace kaldi
