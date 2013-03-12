// fstext/deterministic-fst-inl.h

// Copyright 2011-2012 Gilles Boulianne  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_FSTEXT_DETERMINISTIC_FST_INL_H_
#define KALDI_FSTEXT_DETERMINISTIC_FST_INL_H_
#include "base/kaldi-common.h"
#include "fstext/fstext-utils.h"


namespace fst {
// Do not include this file directly.  It is included by deterministic-fst.h.

template<class Arc>
typename Arc::StateId
BackoffDeterministicOnDemandFst<Arc>::GetBackoffState(StateId s,
                                                      Weight *w) {
  ArcIterator<Fst<Arc> > aiter(fst_, s);
  if (aiter.Done()) // no arcs.
    return kNoStateId;
  const Arc &arc = aiter.Value();
  if (arc.ilabel == 0) {
    *w = arc.weight;
    return arc.nextstate;
  } else {
    return kNoStateId;
  }
}

template<class Arc>
typename Arc::Weight BackoffDeterministicOnDemandFst<Arc>::Final(StateId state) {
  Weight w = fst_.Final(state);
  if (w != Weight::Zero()) return w;
  Weight backoff_w;
  StateId backoff_state = GetBackoffState(state, &backoff_w);
  if (backoff_state == kNoStateId) return Weight::Zero();
  else return Times(backoff_w, this->Final(backoff_state));
}

template<class Arc>
BackoffDeterministicOnDemandFst<Arc>::BackoffDeterministicOnDemandFst(
    const Fst<Arc> &fst): fst_(fst) {
#ifdef KALDI_PARANOID
  KALDI_ASSERT(fst_.Properties(kILabelSorted|kIDeterministic, true) ==
               (kILabelSorted|kIDeterministic) &&
               "Input FST is not i-label sorted and deterministic.");
#endif
}

template<class Arc>
bool BackoffDeterministicOnDemandFst<Arc>::GetArc(
    StateId s, Label ilabel, Arc *oarc) {
  KALDI_ASSERT(ilabel != 0); //  We don't allow GetArc for epsilon.

  SortedMatcher<Fst<Arc> > sm(fst_, MATCH_INPUT, 1);
  sm.SetState(s);
  if (sm.Find(ilabel)) {
    const Arc &arc = sm.Value();
    *oarc = arc;
    return true;
  } else {
    Weight backoff_w;
    StateId backoff_state = GetBackoffState(s, &backoff_w);
    if (backoff_state == kNoStateId) return false;
    if (!this->GetArc(backoff_state, ilabel, oarc)) return false;
    oarc->weight = Times(oarc->weight, backoff_w);
    return true;
  }
}

template<class Arc>
ComposeDeterministicOnDemandFst<Arc>::ComposeDeterministicOnDemandFst(
    DeterministicOnDemandFst<Arc> *fst1,
    DeterministicOnDemandFst<Arc> *fst2): fst1_(fst1), fst2_(fst2) {
  KALDI_ASSERT(fst1 != NULL && fst2 != NULL);
  if (fst1_->Start() == -1 || fst2_->Start() == -1) {
    start_state_ = -1;
    next_state_ = 0; // actually we don't care about this value.
  } else {
    start_state_ = 0;
    std::pair<StateId,StateId> start_pair(fst1_->Start(), fst2_->Start());
    state_map_[start_pair] = start_state_;
    state_vec_.push_back(start_pair);
    next_state_ = 1;
  }
}

template<class Arc>
typename Arc::Weight ComposeDeterministicOnDemandFst<Arc>::Final(StateId s) {
  KALDI_ASSERT(s < static_cast<StateId>(state_vec_.size()));
  const std::pair<StateId, StateId> &pr (state_vec_[s]);
  return Times(fst1_->Final(pr.first), fst2_->Final(pr.second));
}

template<class Arc>
bool ComposeDeterministicOnDemandFst<Arc>::GetArc(StateId s, Label ilabel,
                                                  Arc *oarc) {
  typedef typename MapType::iterator IterType;
  KALDI_ASSERT(ilabel != 0);
  KALDI_ASSERT(s < static_cast<StateId>(state_vec_.size()));
  const std::pair<StateId, StateId> pr (state_vec_[s]);
  
  Arc arc1;
  if (!fst1_->GetArc(pr.first, ilabel, &arc1)) return false;
  if (arc1.olabel == 0) { // There is no output label on the
    // arc, so only the first state changes.
    std::pair<const std::pair<StateId, StateId>, StateId> new_value(
        std::pair<StateId, StateId>(arc1.nextstate, pr.second),
        next_state_);
    
    std::pair<IterType, bool> result = state_map_.insert(new_value);    
    oarc->ilabel = ilabel;
    oarc->olabel = 0;
    oarc->nextstate = result.first->second;
    oarc->weight = arc1.weight;
    if (result.second == true) { // was inserted
      next_state_++;
      const std::pair<StateId, StateId> &new_pair (new_value.first);
      state_vec_.push_back(new_pair);
    }
    return true;
  }
  // There is an output label, so we need to traverse an arc on the
  // second fst also.
  Arc arc2;
  if (!fst2_->GetArc(pr.second, arc1.olabel, &arc2)) return false;
  std::pair<const std::pair<StateId, StateId>, StateId> new_value(
      std::pair<StateId, StateId>(arc1.nextstate, arc2.nextstate),
      next_state_);
  std::pair<IterType, bool> result =
      state_map_.insert(new_value);
  oarc->ilabel = ilabel;
  oarc->olabel = arc2.olabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Times(arc1.weight, arc2.weight);
  if (result.second == true) { // was inserted
    next_state_++;
    const std::pair<StateId, StateId> &new_pair (new_value.first);
    state_vec_.push_back(new_pair);
  }
  return true;
}

template<class Arc>
inline size_t CacheDeterministicOnDemandFst<Arc>::GetIndex(
    StateId src_state, Label ilabel) {
  const StateId p1 = 26597, p2 = 50329; // these are two
  // values that I drew at random from a table of primes.
  // note: num_cached_arcs_ > 0.

  // We cast to size_t before the modulus, to ensure the
  // result is positive.
  return static_cast<size_t>(src_state * p1 + ilabel * p2) %
      static_cast<size_t>(num_cached_arcs_);
}

template<class Arc>
CacheDeterministicOnDemandFst<Arc>::CacheDeterministicOnDemandFst(
    DeterministicOnDemandFst<Arc> *fst,
    StateId num_cached_arcs): fst_(fst),
                              num_cached_arcs_(num_cached_arcs),
                              cached_arcs_(num_cached_arcs) {
  KALDI_ASSERT(num_cached_arcs > 0);
  for (StateId i = 0; i < num_cached_arcs; i++)
    cached_arcs_[i].first = kNoStateId; // Invalidate all elements of the cache.
}
      
template<class Arc>
bool CacheDeterministicOnDemandFst<Arc>::GetArc(StateId s, Label ilabel,
                                                Arc *oarc) {
  // Note: we don't cache anything in case a requested arc does not exist.
  // In the uses that we imagine this will be put to, essentially all the
  // requested arcs will exist.  This only affects efficiency.
  KALDI_ASSERT(s >= 0 && ilabel != 0);
  size_t index = this->GetIndex(s, ilabel);
  if (cached_arcs_[index].first == s &&
      cached_arcs_[index].second.ilabel == ilabel) {
    *oarc = cached_arcs_[index].second;
    return true;
  } else {
    Arc arc;
    if (fst_->GetArc(s, ilabel, &arc)) {
      cached_arcs_[index].first = s;
      cached_arcs_[index].second = arc;
      *oarc = arc;
      return true;
    } else {
      return false;
    }
  }  
}



} // end namespace fst


#endif