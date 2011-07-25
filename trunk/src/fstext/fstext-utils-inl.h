// fstext/fstext-utils-inl.h

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

#ifndef KALDI_FSTEXT_FSTEXT_UTILS_INL_H_
#define KALDI_FSTEXT_FSTEXT_UTILS_INL_H_
#include <cstring>
#include "base/kaldi-common.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"
#include "util/kaldi-io.h"
#include "fstext/factor.h"
#include "fstext/pre-determinize.h"
#include "fstext/determinize-star.h"
#include "fstext/make-stochastic.h"
#include "fstext/reorder.h"

namespace fst {



template<class Arc>
typename Arc::Label HighestNumberedOutputSymbol(const Fst<Arc> &fst) {
  typename Arc::Label ans = 0;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const Arc &arc = aiter.Value();
      ans = std::max(ans, arc.olabel);
    }
  }
  return ans;
}

template<class Arc>
typename Arc::Label HighestNumberedInputSymbol(const Fst<Arc> &fst) {
  typename Arc::Label ans = 0;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const Arc &arc = aiter.Value();
      ans = std::max(ans, arc.ilabel);
    }
  }
  return ans;
}


template<class Arc, class I>
void GetOutputSymbols(const Fst<Arc> &fst,
                      bool include_eps,
                      vector<I> *symbols) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  std::set<I> all_syms;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const Arc &arc = aiter.Value();
      all_syms.insert(arc.olabel);
    }
  }

  // Remove epsilon, if instructed.
  if (!include_eps && !all_syms.empty() && all_syms.front() == 0)
    all_syms.erase(0);
  assert(symbols != NULL);
  kaldi::CopySetToVector(all_syms, symbols);
}

template<class Arc, class I>
void GetInputSymbols(const Fst<Arc> &fst,
                     bool include_eps,
                     vector<I> *symbols) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  std::set<I> all_syms;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const Arc &arc = aiter.Value();
      all_syms.insert(arc.ilabel);
    }
  }
  // Remove epsilon, if instructed.
  if (!include_eps && !all_syms.empty() && *(all_syms.begin()) == 0)
    all_syms.erase(0);
  assert(symbols != NULL);
  kaldi::CopySetToVector(all_syms, symbols);
}


template<class Arc, class I>
void RemoveSomeInputSymbols(const vector<I> &to_remove,
                            MutableFst<Arc> *fst) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  RemoveSomeInputSymbolsMapper<Arc, I> mapper(to_remove);
  Map(fst, mapper);
}

template<class Arc, class I>
class MapInputSymbolsMapper {
 public:
  Arc operator ()(const Arc &arc_in) {
    Arc ans = arc_in;
    if (ans.ilabel > 0 &&
       ans.ilabel < static_cast<typename Arc::Label>((*symbol_mapping_).size()))
      ans.ilabel = (*symbol_mapping_)[ans.ilabel];
    return ans;
  }
  MapFinalAction FinalAction() { return MAP_NO_SUPERFINAL; }
  MapSymbolsAction InputSymbolsAction() { return MAP_CLEAR_SYMBOLS; }
  MapSymbolsAction OutputSymbolsAction() { return MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const {  // Not tested.
    bool remove_epsilons = (symbol_mapping_->size() > 0 && (*symbol_mapping_)[0] != 0);
    bool add_epsilons = (symbol_mapping_->size() > 1 &&
                         *std::min_element(symbol_mapping_->begin()+1, symbol_mapping_->end()) == 0);

    // remove the following as we don't know now if any of them are true.
    uint64 props_to_remove = kAcceptor|kNotAcceptor|kIDeterministic|kNonIDeterministic|
        kILabelSorted|kNotILabelSorted;
    if (remove_epsilons) props_to_remove |= kEpsilons|kIEpsilons;
    if (add_epsilons) props_to_remove |=  kNoEpsilons|kNoIEpsilons;
    uint64 props_to_add = 0;
    if (remove_epsilons && !add_epsilons) props_to_add |= kNoEpsilons|kNoIEpsilons;
    return (props & ~props_to_remove) | props_to_add;
  }
  // initialize with copy = false only if the "to_remove" argument will not be deleted
  // in the lifetime of this object.
  MapInputSymbolsMapper(const vector<I> &to_remove, bool copy) {
    KALDI_ASSERT_IS_INTEGER_TYPE(I);
    if (copy) symbol_mapping_ = new vector<I> (to_remove);
    else symbol_mapping_ = &to_remove;
    owned = copy;
  }
  ~MapInputSymbolsMapper() { if (owned && symbol_mapping_ != NULL) delete symbol_mapping_; }
 private:
  bool owned;
  const vector<I> *symbol_mapping_;
};

template<class Arc, class I>
void MapInputSymbols(const vector<I> &symbol_mapping,
                     MutableFst<Arc> *fst) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  // false == don't copy the "symbol_mapping", retain pointer--
  // safe since short-lived object.
  MapInputSymbolsMapper<Arc, I> mapper(symbol_mapping, false);
  Map(fst, mapper);
}

template<class Arc, class I>
bool GetLinearSymbolSequence(const Fst<Arc> &fst,
                             vector<I> *isymbols_out,
                             vector<I> *osymbols_out,
                             typename Arc::Weight *tot_weight_out) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  Weight tot_weight = Weight::One();
  vector<I> ilabel_seq;
  vector<I> olabel_seq;

  StateId cur_state = fst.Start();
  if (cur_state == kNoStateId) {  // empty sequence.
    if (isymbols_out != NULL) isymbols_out->clear();
    if (osymbols_out != NULL) osymbols_out->clear();
    if (tot_weight_out != NULL) *tot_weight_out = Weight::Zero();
    return true;
  }
  while (1) {
    Weight w = fst.Final(cur_state);
    if (w != Weight::Zero()) {  // is final..
      tot_weight = Times(w, tot_weight);
      if (fst.NumArcs(cur_state) != 0) return false;
      if (isymbols_out != NULL) *isymbols_out = ilabel_seq;
      if (osymbols_out != NULL) *osymbols_out = olabel_seq;
      if (tot_weight_out != NULL) *tot_weight_out = tot_weight;
      return true;
    } else {
      if (fst.NumArcs(cur_state) != 1) return false;

      ArcIterator<Fst<Arc> > iter(fst, cur_state);  // get the only arc.
      const Arc &arc = iter.Value();
      tot_weight = Times(arc.weight, tot_weight);
      if (arc.ilabel != 0) ilabel_seq.push_back(arc.ilabel);
      if (arc.olabel != 0) olabel_seq.push_back(arc.olabel);
      cur_state = arc.nextstate;
    }
  }
}

template<class Arc, class I>
void MakeLinearAcceptorWithAlternatives(const vector<vector<I> > &labels,
                                        MutableFst<Arc> *ofst) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  ofst->DeleteStates();
  StateId cur_state = ofst->AddState();
  ofst->SetStart(cur_state);
  for (size_t i = 0; i < labels.size(); i++) {
    assert(labels[i].size() != NULL);
    StateId next_state = ofst->AddState();
    for (size_t j = 0; j < labels[i].size(); j++) {
      Arc arc(labels[i], labels[i], Weight::One(), next_state);
      ofst->AddArc(cur_state, arc);
    }
    cur_state = next_state;
  }
  ofst->SetFinal(cur_state, Weight::One());
}

template<class Arc, class I>
void MakeLinearAcceptor(const vector<I> &labels, MutableFst<Arc> *ofst) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  ofst->DeleteStates();
  StateId cur_state = ofst->AddState();
  ofst->SetStart(cur_state);
  for (size_t i = 0; i < labels.size(); i++) {
    StateId next_state = ofst->AddState();
    Arc arc(labels[i], labels[i], Weight::One(), next_state);
    ofst->AddArc(cur_state, arc);
    cur_state = next_state;
  }
  ofst->SetFinal(cur_state, Weight::One());
}


template<class I>
void GetSymbols(const SymbolTable &symtab,
                bool include_eps,
                vector<I> *syms_out) {
  assert(syms_out != NULL);
  syms_out->clear();
  for (SymbolTableIterator iter(symtab);
      !iter.Done();
      iter.Next()) {
    if (include_eps || iter.Value() != 0) {
      syms_out->push_back(iter.Value());
      assert(syms_out->back() == iter.Value());  // an integer-range thing.
    }
  }
}

template<class Arc>
void SafeDeterminizeWrapper(MutableFst<Arc> *ifst, MutableFst<Arc> *ofst, float delta) {
  typename Arc::Label highest_sym = HighestNumberedInputSymbol(*ifst);
  vector<typename Arc::Label> extra_syms;
  PreDeterminize(ifst,
                 (typename Arc::Label)(highest_sym+1),
                 &extra_syms);
  DeterminizeStar(*ifst, ofst, delta);
  RemoveSomeInputSymbols(extra_syms, ofst);  // remove the extra symbols.
}


template<class Arc>
void SafeDeterminizeMinimizeWrapper(MutableFst<Arc> *ifst, VectorFst<Arc> *ofst, float delta) {
  typename Arc::Label highest_sym = HighestNumberedInputSymbol(*ifst);
  vector<typename Arc::Label> extra_syms;
  PreDeterminize(ifst,
                 (typename Arc::Label)(highest_sym+1),
                 &extra_syms);
  DeterminizeStar(*ifst, ofst, delta);
  RemoveSomeInputSymbols(extra_syms, ofst);  // remove the extra symbols.
  RemoveEpsLocal(ofst);  // this is "safe" and will never hurt.
  MinimizeEncoded(ofst, delta);
}


inline
void DeterminizeStarInLog(VectorFst<StdArc> *fst, float delta, bool *debug_ptr, int max_states) {
  // DeterminizeStarInLog determinizes 'fst' in the log semiring, using
  // the DeterminizeStar algorithm (which also removes epsilons).

  ArcSort(fst, ILabelCompare<StdArc>());  // helps DeterminizeStar to be faster.
  VectorFst<LogArc> *fst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*fst, fst_log);
  VectorFst<StdArc> tmp;
  *fst = tmp;  // make fst empty to free up memory. [actually may make no difference..]
  VectorFst<LogArc> *fst_det_log = new VectorFst<LogArc>;
  DeterminizeStar(*fst_log, fst_det_log, delta, debug_ptr, max_states);
  Cast(*fst_det_log, fst);
  delete fst_log;
  delete fst_det_log;
}

inline
void DeterminizeInLog(VectorFst<StdArc> *fst) {
  // DeterminizeInLog determinizes 'fst' in the log semiring.

  ArcSort(fst, ILabelCompare<StdArc>());  // helps DeterminizeStar to be faster.
  VectorFst<LogArc> *fst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*fst, fst_log);
  VectorFst<StdArc> tmp;
  *fst = tmp;  // make fst empty to free up memory. [actually may make no difference..]
  VectorFst<LogArc> *fst_det_log = new VectorFst<LogArc>;
  Determinize(*fst_log, fst_det_log);
  Cast(*fst_det_log, fst);
  delete fst_log;
  delete fst_det_log;
}



// make it inline to avoid having to put it in a .cc file.
// destructive algorithm (changes ifst as well as ofst).
inline
void SafeDeterminizeMinimizeWrapperInLog(VectorFst<StdArc> *ifst, VectorFst<StdArc> *ofst, float delta) {
  VectorFst<LogArc> *ifst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*ifst, ifst_log);
  VectorFst<LogArc> *ofst_log = new VectorFst<LogArc>;
  SafeDeterminizeWrapper(ifst_log, ofst_log, delta);
  Cast(*ofst_log, ofst);
  delete ifst_log;
  delete ofst_log;
  RemoveEpsLocal(ofst);  // this is "safe" and will never hurt.  Do this in tropical, which is important.
  MinimizeEncoded(ofst, delta);  // Non-deterministic minimization will fail in log semiring so do it with StdARc.
}

inline
void SafeDeterminizeWrapperInLog(VectorFst<StdArc> *ifst, VectorFst<StdArc> *ofst, float delta) {
  VectorFst<LogArc> *ifst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*ifst, ifst_log);
  VectorFst<LogArc> *ofst_log = new VectorFst<LogArc>;
  SafeDeterminizeWrapper(ifst_log, ofst_log, delta);
  Cast(*ofst_log, ofst);
  delete ifst_log;
  delete ofst_log;
}



template<class Arc>
void RemoveWeights(MutableFst<Arc> *ifst) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  for (StateIterator<MutableFst<Arc> > siter(*ifst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> >  aiter(ifst, s); !aiter.Done(); aiter.Next()) {
      Arc arc(aiter.Value());
      arc.weight = Weight::One();
      aiter.SetValue(arc);
    }
    if (ifst->Final(s) != Weight::Zero())
      ifst->SetFinal(s, Weight::One());
  }
  ifst->SetProperties(kUnweighted, kUnweighted);
}

// Used in PrecedingInputSymbolsAreSame (non-functor version), and
// similar routines.
template<class T> struct IdentityFunction {
  typedef T Arg;
  typedef T Result;
  T operator () (const T &t) const { return t; }
};

template<class Arc>
bool PrecedingInputSymbolsAreSame(bool start_is_epsilon, const Fst<Arc> &fst) {
  IdentityFunction<typename Arc::Label> f;
  return PrecedingInputSymbolsAreSameClass(start_is_epsilon, fst, f);
}

template<class Arc, class F> // F is functor type from labels to classes.
bool PrecedingInputSymbolsAreSameClass(bool start_is_epsilon, const Fst<Arc> &fst, const F &f) {
  typedef typename Arc::Label Label;
  typedef typename F::Result ClassType;
  typedef typename Arc::StateId StateId;
  vector<ClassType> classes;
  ClassType noClass = f(kNoLabel);

  if (start_is_epsilon) {
    StateId start_state = fst.Start();
    if (start_state < 0 || start_state == kNoStateId)
      return true;  // empty fst-- doesn't matter.
    classes.resize(start_state+1, noClass);
    classes[start_state] = 0;
  }

  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (classes.size() <= arc.nextstate)
        classes.resize(arc.nextstate+1, noClass);
      if (classes[arc.nextstate] == noClass)
        classes[arc.nextstate] = f(arc.ilabel);
      else
        if (classes[arc.nextstate] != f(arc.ilabel))
          return false;
    }
  }
  return true;
}

template<class Arc>
bool FollowingInputSymbolsAreSame(bool end_is_epsilon, const Fst<Arc> &fst) {
  IdentityFunction<typename Arc::Label> f;
  return FollowingInputSymbolsAreSameClass(end_is_epsilon, fst, f);
}


template<class Arc, class F>
bool FollowingInputSymbolsAreSameClass(bool end_is_epsilon, const Fst<Arc> &fst, const F &f) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename F::Result ClassType;
  const ClassType noClass = f(kNoLabel), epsClass = f(0);
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    ClassType c = noClass;
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (c == noClass)
        c = f(arc.ilabel);
      else
        if (c != f(arc.ilabel))
          return false;
    }
    if (end_is_epsilon && c != noClass &&
       c != epsClass && fst.Final(s) != Weight::Zero())
      return false;
  }
  return true;
}

template<class Arc>
void MakePrecedingInputSymbolsSame(bool start_is_epsilon, MutableFst<Arc> *fst) {
  IdentityFunction<typename Arc::Label> f;
  MakePrecedingInputSymbolsSameClass(start_is_epsilon, fst, f);
}

template<class Arc, class F>
void MakePrecedingInputSymbolsSameClass(bool start_is_epsilon, MutableFst<Arc> *fst, const F &f) {
  typedef typename F::Result ClassType;
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  vector<ClassType> classes;
  ClassType noClass = f(kNoLabel);
  ClassType epsClass = f(0);
  if (start_is_epsilon) {  // treat having-start-state as epsilon in-transition.
    StateId start_state = fst->Start();
    if (start_state < 0 || start_state == kNoStateId) // empty FST.
      return;
    classes.resize(start_state+1, noClass);
    classes[start_state] = epsClass;
  }

  // Find bad states (states with multiple input-symbols into them).
  std::set<StateId> bad_states;  // states that we need to change.
  for (StateIterator<Fst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (classes.size() <= static_cast<size_t>(arc.nextstate))
        classes.resize(arc.nextstate+1, noClass);
      if (classes[arc.nextstate] == noClass)
        classes[arc.nextstate] = f(arc.ilabel);
      else
        if (classes[arc.nextstate] != f(arc.ilabel))
          bad_states.insert(arc.nextstate);
    }
  }
  if (bad_states.empty()) return;  // Nothing to do.
  kaldi::ConstIntegerSet<StateId> bad_states_ciset(bad_states);  // faster lookup.

  // Work out list of arcs we have to change as (state, arc-offset).
  // Can't do the actual changes in this pass, since we have to add new
  // states which invalidates the iterators.
  vector<pair<StateId, size_t> > arcs_to_change;
  for (StateIterator<Fst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0 &&
         bad_states_ciset.count(arc.nextstate) != 0)
        arcs_to_change.push_back(std::make_pair(s, aiter.Position()));
    }
  }
  assert(!arcs_to_change.empty());  // since !bad_states.empty().

  std::map<pair<StateId, ClassType>, StateId> state_map;
  // state_map is a map from (bad-state, input-symbol-class) to dummy-state.

  for (size_t i = 0; i < arcs_to_change.size(); i++) {
    StateId s = arcs_to_change[i].first;
    ArcIterator<MutableFst<Arc> > aiter(*fst, s);
    aiter.Seek(arcs_to_change[i].second);
    Arc arc = aiter.Value();

    // Transition is non-eps transition to "bad" state.  Introduce new state (or find
    // existing one).
    pair<StateId, ClassType> p(arc.nextstate, f(arc.ilabel));
    if (state_map.count(p) == 0) {
      StateId newstate = state_map[p] = fst->AddState();
      fst->AddArc(newstate, Arc(0, 0, Weight::One(), arc.nextstate));
    }
    StateId dst_state = state_map[p];
    arc.nextstate = dst_state;

    // Initialize the MutableArcIterator only now, as the call to NewState()
    // may have invalidated the first arc iterator.
    MutableArcIterator<MutableFst<Arc> > maiter(fst, s);
    maiter.Seek(arcs_to_change[i].second);
    maiter.SetValue(arc);
  }
}

template<class Arc>
void MakeFollowingInputSymbolsSame(bool end_is_epsilon, MutableFst<Arc> *fst) {
  IdentityFunction<typename Arc::Label> f;
  MakeFollowingInputSymbolsSameClass(end_is_epsilon, fst, f);
}

template<class Arc, class F>
void MakeFollowingInputSymbolsSameClass(bool end_is_epsilon, MutableFst<Arc> *fst, const F &f) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename F::Result ClassType;
  vector<StateId> bad_states;
  ClassType noClass = f(kNoLabel);
  ClassType epsClass = f(0);
  for (StateIterator<Fst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    ClassType c = noClass;
    bool bad = false;
    for (ArcIterator<Fst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (c == noClass)
        c = f(arc.ilabel);
      else
        if (c != f(arc.ilabel)) {
          bad = true;
          break;
        }
    }
    if (end_is_epsilon && c != noClass &&
       c != epsClass && fst->Final(s) != Weight::Zero())
      bad = true;
    if (bad)
      bad_states.push_back(s);
  }
  vector<Arc> my_arcs;
  for (size_t i = 0; i < bad_states.size(); i++) {
    StateId s = bad_states[i];
    my_arcs.clear();
    for (ArcIterator<MutableFst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next())
      my_arcs.push_back(aiter.Value());

    for (size_t j = 0; j < my_arcs.size(); j++) {
      Arc &arc = my_arcs[j];
      if (arc.ilabel != 0) {
        StateId newstate = fst->AddState();
        // Create a new state for each non-eps arc in original FST, out of each bad state.
        // Not as optimal as it could be, but does avoid some complicated weight-pushing
        // issues in which, to maintain stochasticity, we would have to know which semiring
        // we want to maintain stochasticity in.
        fst->AddArc(newstate, Arc(arc.ilabel, 0, Weight::One(), arc.nextstate));
        MutableArcIterator<MutableFst<Arc> > maiter(fst, s);
        maiter.Seek(j);
        maiter.SetValue(Arc(0, arc.olabel, arc.weight, newstate));
      }
    }
  }
}


template<class Arc>
VectorFst<Arc>* MakeLoopFst(const vector<const ExpandedFst<Arc> *> &fsts) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  VectorFst<Arc> *ans = new VectorFst<Arc>;
  StateId loop_state = ans->AddState();  // = 0.
  ans->SetStart(loop_state);
  ans->SetFinal(loop_state, Weight::One());

  for (Label i = 0; i < static_cast<Label>(fsts.size()); i++) {
    const ExpandedFst<Arc> *fst = fsts[i];
    if (fst == NULL) continue;
    assert(fst->Properties(kAcceptor, true) == kAcceptor);  // expect acceptor.

    StateId fst_num_states = fst->NumStates();
    StateId fst_start_state = fst->Start();

    if (fst_start_state == kNoStateId)
      continue;  // empty fst.

    bool share_start_state =
        fst->Properties(kInitialAcyclic, true) == kInitialAcyclic
        && fst->NumArcs(fst_start_state) == 1
        && fst->Final(fst_start_state) == Weight::Zero();

    vector<StateId> state_map(fst_num_states);  // fst state -> ans state
    for (StateId s = 0; s < fst_num_states; s++) {
      if (s == fst_start_state && share_start_state) state_map[s] = loop_state;
      else state_map[s] = ans->AddState();
    }
    if (!share_start_state)
      ans->AddArc(0, Arc(0, i, Weight::One(), state_map[fst_start_state]));
    for (StateId s = 0; s < fst_num_states; s++) {
      // Add arcs out of state s.
      for (ArcIterator<ExpandedFst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        Label olabel = (s == fst_start_state && share_start_state ? i : 0);
        ans->AddArc(state_map[s],
                    Arc(arc.ilabel, olabel, arc.weight, state_map[arc.nextstate]));
      }
      if (fst->Final(s) != Weight::Zero()) {
        assert(!(s == fst_start_state && share_start_state));
        ans->AddArc(state_map[s], Arc(0, 0, fst->Final(s), loop_state));
      }
    }
  }
  return ans;
}


template<class Arc>
void ClearSymbols(bool clear_input,
                  bool clear_output,
                  MutableFst<Arc> *fst) {
  for (StateIterator<MutableFst<Arc> > siter(*fst);
       !siter.Done();
       siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
         !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      bool change = false;
      if (clear_input && arc.ilabel != 0) {
        arc.ilabel = 0;
        change = true;
      }
      if (clear_output && arc.olabel != 0) {
        arc.olabel = 0;
        change = true;
      }
      if (change) {
        aiter.SetValue(arc);
      }
    }
  }
}


template<class Arc>
void ApplyProbabilityScale(float scale, MutableFst<Arc> *fst) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  for (StateIterator<MutableFst<Arc> > siter(*fst);
       !siter.Done();
       siter.Next()) {
    StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
        !aiter.Done();
        aiter.Next()) {
      Arc arc = aiter.Value();
      arc.weight = Weight(arc.weight.Value() * scale);
      aiter.SetValue(arc);
    }
    if (fst->Final(s) != Weight::Zero())
      fst->SetFinal(s, Weight(fst->Final(s).Value() * scale));
  }
}



void Optimize(VectorFst<StdArc> *fst,
              const OptimizeConfig &cfg) {
  typedef StdArc::Label Label;
  KALDI_VLOG(1) <<  "Optimize: about to determinize.";
  {  // Determinize.
    VectorFst<StdArc> det_fst;
    if (cfg.maintain_log_stochasticity) SafeDeterminizeWrapperInLog(fst, &det_fst, cfg.delta);
    else SafeDeterminizeWrapper(fst, &det_fst, cfg.delta);
    *fst = det_fst;
  }

  if (cfg.maintain_log_stochasticity) RemoveEpsLocalSpecial(fst);
  else RemoveEpsLocal(fst);

  if (cfg.push_weights || cfg.push_labels) {  // need to do some kind of pushing...
    KALDI_VLOG(1) <<  "Optimize: about to push.";
    if (cfg.push_weights &&  cfg.push_in_log) {
      VectorFst<LogArc> *log_fst = new VectorFst<LogArc>;
      Cast(*fst, log_fst);
      VectorFst<LogArc> *log_fst_pushed = new VectorFst<LogArc>;

      Push<LogArc, REWEIGHT_TO_INITIAL>
           (*log_fst, log_fst_pushed,
            (cfg.push_weights?kPushWeights:0)|(cfg.push_labels?kPushLabels:0),
            cfg.delta);

      Cast(*log_fst_pushed, fst);
      delete log_fst;
      delete log_fst_pushed;
    } else {
      VectorFst<StdArc> fst_pushed;
      Push<StdArc, REWEIGHT_TO_INITIAL>
          (*fst, &fst_pushed,
           (cfg.push_weights?kPushWeights:0)|(cfg.push_labels?kPushLabels:0),
           cfg.delta);
      *fst = fst_pushed;
    }
  }
  KALDI_VLOG(1) <<  "Optimize: about to minimize.";
  MinimizeEncoded(fst, cfg.delta);

  // now do DFS-order.
  KALDI_VLOG(1) << "Optimize: about to sort arcs by weight.";
  WeightArcSort(fst);
  KALDI_VLOG(1) << "Optimize: about to dfs order.";
  VectorFst<StdArc> ordered;
  DfsReorder(*fst, &ordered);
  *fst = ordered;
}



// return arc-offset of self-loop with ilabel (or -1 if none exists).
// if more than one such self-loop, pick first one.
template<class Arc>
ssize_t FindSelfLoopWithILabel(const Fst<Arc> &fst, typename Arc::StateId s) {
  for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next())
    if (aiter.Value().nextstate == s
       && aiter.Value().ilabel != 0) return static_cast<ssize_t>(aiter.Position());
  return static_cast<ssize_t>(-1);
}


template<class Arc>
bool EqualAlign(const Fst<Arc> &ifst,
                typename Arc::StateId length,
                int rand_seed,
                MutableFst<Arc> *ofst) {
  srand(rand_seed);
  assert(ofst->NumStates() == 0);  // make sure ofst empty.
  // make sure all states can reach final-state (or this algorithm may enter
  // infinite loop.
  assert(ifst.Properties(kCoAccessible, true) == kCoAccessible);

  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  if (ifst.Start() == kNoStateId) return false;
  // First select path through ifst.
  vector<StateId> path;
  vector<size_t> arc_offsets;  // arc taken out of each state.

  path.push_back(ifst.Start());
  StateId num_ilabels = 0;
  while (1) {
    // Select either an arc or final-prob.
    StateId s = path.back();
    size_t num_arcs = ifst.NumArcs(s);
    size_t num_arcs_tot = num_arcs;
    if (ifst.Final(s) != Weight::Zero()) num_arcs_tot++;
    // kaldi::RandInt is a bit like rand(), but gets around situations
    // where RAND_MAX is very small.
    // Change this to rand() % num_arcs_tot if compile issues arise
    size_t arc_offset = static_cast<size_t>(kaldi::RandInt(0, num_arcs_tot-1));

    if (arc_offset < num_arcs) {  // an actual arc.
      ArcIterator<Fst<Arc> > aiter(ifst, s);
      aiter.Seek(arc_offset);
      const Arc &arc = aiter.Value();
      if (arc.nextstate == s) {
        continue;  // don't take this self-loop arc
      } else {
        arc_offsets.push_back(arc_offset);
        path.push_back(arc.nextstate);
        if (arc.ilabel != 0) num_ilabels++;
      }
    } else {
      break;  // Chose final-prob.
    }
  }

  if (num_ilabels > length) return false;  // can't make it shorter by adding self-loops!.

  StateId num_self_loops = 0;
  vector<ssize_t> self_loop_offsets(path.size());
  for (size_t i = 0; i < path.size(); i++)
    if ( (self_loop_offsets[i] = FindSelfLoopWithILabel(ifst, path[i]))
         != static_cast<ssize_t>(-1) )
      num_self_loops++;

  if (num_self_loops == 0
     && num_ilabels < length) return false;  // no self-loops to make it longer.

  StateId num_extra = length - num_ilabels;  // Number of self-loops we need.

  StateId min_num_loops = 0;
  if (num_extra != 0) min_num_loops = num_extra / num_self_loops;  // prevent div by zero.
  StateId num_with_one_more_loop = num_extra - (min_num_loops*num_self_loops);
  assert(num_with_one_more_loop < num_self_loops);

  ofst->AddState();
  ofst->SetStart(0);
  StateId cur_state = 0;
  StateId counter = 0;  // tell us when we should stop adding one more loop.
  for (size_t i = 0; i < path.size(); i++) {
    // First, add any self-loops that are necessary.
    StateId num_loops = 0;
    if (self_loop_offsets[i] != static_cast<ssize_t>(-1)) {
      num_loops = min_num_loops + (counter < num_with_one_more_loop ? 1 : 0);
      counter++;
    }
    for (StateId j = 0; j < num_loops; j++) {
      ArcIterator<Fst<Arc> > aiter(ifst, path[i]);
      aiter.Seek(self_loop_offsets[i]);
      Arc arc = aiter.Value();
      assert(arc.nextstate == path[i]
             && arc.ilabel != 0);  // make sure self-loop with ilabel.
      StateId next_state = ofst->AddState();
      ofst->AddArc(cur_state, Arc(arc.ilabel, arc.olabel, arc.weight, next_state));
      cur_state = next_state;
    }
    if (i+1 < path.size()) {  // add forward transition.
      ArcIterator<Fst<Arc> > aiter(ifst, path[i]);
      aiter.Seek(arc_offsets[i]);
      Arc arc = aiter.Value();
      assert(arc.nextstate == path[i+1]);
      StateId next_state = ofst->AddState();
      ofst->AddArc(cur_state, Arc(arc.ilabel, arc.olabel, arc.weight, next_state));
      cur_state = next_state;
    } else {  // add final-prob.
      Weight weight = ifst.Final(path[i]);
      assert(weight != Weight::Zero());
      ofst->SetFinal(cur_state, weight);
    }
  }
  return true;
}


// This function identifies two types of useless arcs:
// those where arc A and arc B both go from state X to
// state Y with the same input symbol (remove the one
// with smaller probability, or an arbitrary one if they
// are the same); and those where A is an arc from state X
// to state X, with epsilon input symbol [remove A].
// Only works for tropical (not log) semiring as it uses
// NaturalLess.
template<class Arc>
void RemoveUselessArcs(MutableFst<Arc> *fst) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  NaturalLess<Weight> nl;
  StateId non_coacc_state = kNoStateId;
  size_t num_arcs_removed = 0, tot_arcs = 0;
  for (StateIterator<MutableFst<Arc> > siter(*fst);
      !siter.Done();
      siter.Next()) {
    vector<size_t> arcs_to_delete;
    vector<Arc> arcs;
    // pair2arclist lets us look up the arcs
    std::map<pair<Label, StateId>, vector<size_t> > pair2arclist;
    StateId state = siter.Value();
    for (ArcIterator<MutableFst<Arc> > aiter(*fst, state);
        !aiter.Done();
        aiter.Next()) {
      size_t pos = arcs.size();
      const Arc &arc = aiter.Value();
      arcs.push_back(arc);
      pair2arclist[std::make_pair(arc.ilabel, arc.nextstate)].push_back(pos);
    }
    typename std::map<pair<Label, StateId>, vector<size_t> >::iterator
        iter = pair2arclist.begin(), end = pair2arclist.end();
    for (; iter!= end; ++iter) {
      const vector<size_t> &poslist = iter->second;
      if (poslist.size() > 1) {  // >1 arc with same ilabel, dest-state
        size_t best_pos = poslist[0];
        Weight best_weight = arcs[best_pos].weight;
        for (size_t j = 1; j < poslist.size(); j++) {
          size_t pos = poslist[j];
          Weight this_weight = arcs[pos].weight;
          if (nl(this_weight, best_weight)) {  // NaturalLess seems to be somehow
            // "backwards".
            best_weight = this_weight;  // found a better one.
            best_pos = pos;
          }
        }
        for (size_t j = 0; j < poslist.size(); j++)
          if (poslist[j] != best_pos)
            arcs_to_delete.push_back(poslist[j]);
      } else {
        assert(poslist.size() == 1);
        size_t pos = poslist[0];
        Arc &arc = arcs[pos];
        if (arc.ilabel == 0 && arc.nextstate == state)
          arcs_to_delete.push_back(pos);
      }
    }
    tot_arcs += arcs.size();
    if (arcs_to_delete.size() != 0) {
      num_arcs_removed += arcs_to_delete.size();
      if (non_coacc_state == kNoStateId)
        non_coacc_state = fst->AddState();
      MutableArcIterator<MutableFst<Arc> > maiter(fst, state);
      for (size_t j = 0; j < arcs_to_delete.size(); j++) {
        size_t pos = arcs_to_delete[j];
        maiter.Seek(pos);
        arcs[pos].nextstate = non_coacc_state;
        maiter.SetValue(arcs[pos]);
      }
    }
  }
  if (non_coacc_state != kNoStateId)
    Connect(fst);
  KALDI_VLOG(1) << "removed " << num_arcs_removed << " of " << tot_arcs
                << "arcs.";
}





} // namespace fst.

#endif
