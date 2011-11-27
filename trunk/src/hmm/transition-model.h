// hmm/transition-model.h

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

#ifndef KALDI_HMM_TRANSITION_MODEL_H_
#define KALDI_HMM_TRANSITION_MODEL_H_

#include "base/kaldi-common.h"
#include "util/parse-options.h"
#include "tree/context-dep.h"
#include "util/const-integer-set.h"
#include "fst/fst-decl.h" // forward declarations.
#include "hmm/hmm-topology.h"


namespace kaldi {

/// \addtogroup hmm_group
/// @{

// The class TransitionModel is a repository for the transition probabilities.
// It also handles certain integer mappings.
// The basic model is as follows.  Each phone has a HMM topology defined in
// hmm-topology.h.  Each HMM-state of each of these phones has a number of
// transitions (and final-probs) out of it.  Each HMM-state defined in the
// HmmTopology class has an associated "pdf_class".  This gets replaced with
// an actual pdf-id via the tree.  The transition model associates the
// transition probs with the (phone, HMM-state, pdf-id).  We associate with
// each such triple a transition-state.  Each
// transition-state has a number of associated probabilities to estimate;
// this depends on the number of transitions/final-probs in the topology for
// that (phone, HMM-state).  Each probability has an associated transition-index.
// We associate with each (transition-state, transition-index) a unique transition-id.
// Each individual probability estimated by the transition-model is asociated with a
// transition-id.
//
// List of the various types of quantity referred to here and what they mean:
//           phone:  a phone index (1, 2, 3 ...)
//       HMM-state:  a number (0, 1, 2...) that indexes TopologyEntry (see hmm-topology.h)
//          pdf-id:  a number output by the Compute function of ContextDependency (it
//                   indexes pdf's).  Zero-based.
// transition-state:  the states for which we estimate transition probabilities for transitions
//                    out of them.  In some topologies, will map one-to-one with pdf-ids.
//                    One-based, since it appears on FSTs.
// transition-index:  identifier of a transition (or final-prob) in the HMM.  Indexes the
//                    "transitions" vector in HmmTopology::HmmState.  [if it is out of range,
//                    equal to transitions.size(), it refers to the final-prob.]
//                    Zero-based.
//   transition-id:   identifier of a unique parameter of the TransitionModel.
//                    Associated with a (transition-state, transition-index) pair.
//                    One-based, since it appears on FSTs.
//
// List of the possible mappings TransitionModel can do:
//             (phone, HMM-state, pdf-id) -> transition-state
//   (transition-state, transition-index) -> transition-id
//  Reverse mappings:
//                        transition-id -> transition-state
//                        transition-id -> transition-index
//                     transition-state -> phone
//                     transition-state -> HMM-state
//                     transition-state -> pdf-id
//
// The main things the TransitionModel object can do are:
//    Get initialized (need ContextDependency and HmmTopology objects).
//    Read/write.
//    Update [given a vector of counts indexed by transition-id].
//    Do the various integer mappings mentioned above.
//    Get the probability (or log-probability) associated with a particular transition-id.


struct TransitionUpdateConfig {
  BaseFloat floor;
  BaseFloat mincount;
  TransitionUpdateConfig(BaseFloat floor = 0.01,
                         BaseFloat mincount = 5.0):
      floor(floor), mincount(mincount) {}

  void Register (ParseOptions *po) {
    po->Register("transition-floor", &floor, "Floor for transition probabilities");
    po->Register("transition-min-count", &mincount, "Minimum count required to update transitions from a state");
  }
};

class TransitionModel {

 public:
  /// Initialize the object [e.g. at the start of training].
  /// The class keeps a copy of the HmmTopology object, but not
  /// the ContextDependency object.
  TransitionModel(const ContextDependency &ctx_dep,
                  const HmmTopology &hmm_topo);


  /// Constructor that takes no arguments: typically used prior to calling Read.
  TransitionModel() { }

  void Read(std::istream &is, bool binary);  // note, no symbol table: topo object always read/written w/o symbols.
  void Write(std::ostream &os, bool binary) const;


  /// return reference to HMM-topology object.
  const HmmTopology &GetTopo() const { return topo_; }

  /// \name Integer mapping functions
  /// @{

  int32 TripleToTransitionState(int32 phone, int32 hmm_state, int32 pdf) const;
  int32 PairToTransitionId(int32 trans_state, int32 trans_index) const;
  int32 TransitionIdToTransitionState(int32 trans_id) const;
  int32 TransitionIdToTransitionIndex(int32 trans_id) const;
  int32 TransitionStateToPhone(int32 trans_state) const;
  int32 TransitionStateToHmmState(int32 trans_state) const;
  int32 TransitionStateToPdf(int32 trans_state) const;
  int32 SelfLoopOf(int32 trans_state) const;  // returns the self-loop transition-id, or zero if
  // this state doesn't have a self-loop.

  inline int32 TransitionIdToPdf(int32 trans_id) const;
  int32 TransitionIdToPhone(int32 trans_id) const;
  int32 TransitionIdToPdfClass(int32 trans_id) const;
  int32 TransitionIdToHmmState(int32 trans_id) const;

  /// @}

  bool IsFinal(int32 trans_id) const;  // returns true if this trans_id goes to the final state
  // (which is bound to be nonemitting).
  bool IsSelfLoop(int32 trans_id) const;  // return true if this trans_id corresponds to a self-loop.

  /// Returns the total number of transition-ids (note, these are one-based).
  inline int32 NumTransitionIds() const { return id2state_.size()-1; }

  /// Returns the number of transition-indices for a particular transition-state.
  /// Note: "Indices" is the plural of "index".   Index is not the same as "id",
  /// here.  A transition-index is a zero-based offset into the transitions
  /// out of a particular transition state.
  inline int32 NumTransitionIndices(int32 trans_state) const;

  /// Returns the total number of transition-states (note, these are one-based).
  int32 NumTransitionStates() const { return triples_.size(); }

  // NumPdfs() actually returns the highest-numbered pdf we ever saw, plus one.
  // In normal cases this should equal the number of pdfs in the system, but if you
  // initialized this object with fewer than all the phones, and it happens that
  // an unseen phone has the highest-numbered pdf, this might be different.
  int32 NumPdfs() const { return num_pdfs_; }


  /// Returns a sorted, uniq list of phones.
  const std::vector<int32> &GetPhones() const { return topo_.GetPhones(); }

  // Transition-parameter-getting functions:
  BaseFloat GetTransitionProb(int32 trans_id) const;
  BaseFloat GetTransitionLogProb(int32 trans_id) const;

  // The following functions are more specialized functions for getting
  // transition probabilities, that are provided for convenience.

  /// Returns the log-probability of a particular non-self-loop transition
  /// after subtracting the probability mass of the self-loop and renormalizing;
  /// will crash if called on a self-loop.  Specifically:
  /// for non-self-loops it returns the log of that prob divided by (1 minus
  /// self-loop-prob-for-that-state).
  BaseFloat GetTransitionLogProbIgnoringSelfLoops(int32 trans_id) const;

  /// Returns the log-prob of the non-self-loop probability
  /// mass for this transition state. (you can get the self-loop prob, if a self-loop
  /// exists, by/ calling GetTransitionLogProb(SelfLoopOf(trans_state)).
  BaseFloat GetNonSelfLoopLogProb(int32 trans_state) const;

  void Update(const Vector<double> &stats,  // stats are counts/weights, indexed by transition-id.
              const TransitionUpdateConfig &cfg,
              BaseFloat *objf_impr_out,
              BaseFloat *count_out);

  /// Print will print the transition model in a human-readable way, for purposes of human
  /// inspection.  The "occs" are optional (they are indexed by pdf-id).
  void Print(std::ostream &os,
             const std::vector<std::string> &phone_names,
             const Vector<double> *occs = NULL);


  void InitStats(Vector<double> *stats) const { stats->Resize(NumTransitionIds()+1); }

  void Accumulate(BaseFloat prob, int32 trans_id, Vector<double> *stats) const {
    (*stats)(trans_id) += prob;
    // This is trivial and doesn't require class members, but leaves us more open
    // to design changes than doing it manually.
  }

 private:
  void ComputeTriples(const ContextDependency &ctx_dep);  // called from constructor.  initializes triples_.
  void ComputeDerived();  // called from constructor and Read function: computes state2id_ and id2state_.
  void ComputeDerivedOfProbs();  // computes quantities derived from log-probs (currently just
  // non_self_loop_log_probs_; called whenever log-probs change.
  void InitializeProbs();  // called from constructor.
  void Check() const;

  struct Triple {
    int32 phone;
    int32 hmm_state;
    int32 pdf;
    Triple() { }
    Triple(int32 phone, int32 hmm_state, int32 pdf):
        phone(phone), hmm_state(hmm_state), pdf(pdf) { }
    bool operator < (const Triple &other) const {
      if (phone < other.phone) return true;
      else if (phone > other.phone) return false;
      else if (hmm_state < other.hmm_state) return true;
      else if (hmm_state > other.hmm_state) return false;
      else return (pdf < other.pdf);
    }
    bool operator == (const Triple &other) const {
      return (phone == other.phone && hmm_state == other.hmm_state
              && pdf == other.pdf);
    }
  };

  HmmTopology topo_;

  /// Triples indexed by transition state minus one;
  /// the triples are in sorted order which allows us to do the reverse mapping from
  /// triple to transition state
  std::vector<Triple> triples_;

  /// Gives the first transition_id of each transition-state; indexed by
  /// the transition-state.  Array indexed 1..num-transition-states+1 (the last one
  /// is needed so we can know the num-transitions of the last transition-state.
  std::vector<int32> state2id_;

  /// For each transition-id, the corresponding transition
  /// state (indexed by transition-id).
  std::vector<int32> id2state_;

  /// For each transition-id, the corresponding log-prob.  Indexed by transition-id.
  Vector<BaseFloat> log_probs_;

  /// For each transition-state, the log of (1 - self-loop-prob).  Indexed by
  /// transition-state.
  Vector<BaseFloat> non_self_loop_log_probs_;

  /// This is actually one plus the highest-numbered pdf we ever got back from the
  /// tree (but the tree numbers pdfs contiguously from zero so this is the number
  /// of pdfs).
  int32 num_pdfs_;


  DISALLOW_COPY_AND_ASSIGN(TransitionModel);

};

inline int32 TransitionModel::TransitionIdToPdf(int32 trans_id) const {
  // If a lot of time is spent here we may create an extra array
  // to handle this.
  KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32 trans_state = id2state_[trans_id];
  return triples_[trans_state-1].pdf;
}

/// Works out which pdfs might correspond to the given phones.  Will return true
/// if these pdfs correspond *just* to these phones, false if these pdfs are also
/// used by other phones.
/// @param trans_model [in] Transition-model used to work out this information
/// @param phones [in] A sorted, uniq vector that represents a set of phones
/// @param pdfs [out] Will be set to a sorted, uniq list of pdf-ids that correspond
///                   to one of this set of phones.
/// @return  Returns true if all of the pdfs output to "pdfs" correspond to phones from
///          just this set (false if they may be shared with phones outside this set).
bool GetPdfsForPhones(const TransitionModel &trans_model,
                      const std::vector<int32> &phones,
                      std::vector<int32> *pdfs);


/// @}

} // end namespace kaldi


#endif
