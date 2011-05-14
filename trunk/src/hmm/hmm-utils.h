// hmm/hmm-utils.h

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

#ifndef KALDI_HMM_HMM_UTILS_H_
#define KALDI_hmm_HMM_UTILS_H_

#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"

namespace kaldi {


/// \defgroup hmm_group_graph Classes and functions for creating FSTs from HMMs
/// \ingroup hmm_group
/// @{

/// Configuration class for the GetHTransducer() function; see
/// \ref hmm_graph_config for context.
struct HTransducerConfig {
  /// Transition log-prob scale, see \ref hmm_scale.
  /// Note this doesn't apply to self-loops; GetHTransducer() does
  /// not include self-loops.
  BaseFloat trans_prob_scale;

  /// if true, we are constructing time-reversed FST: phone-seqs in ilabel_info
  /// are backwards, and we want to output a backwards version of the HMM
  /// corresponding to each phone.  If reverse == true,
  bool reverse;

  /// This variable is only looked at if reverse == true.  If reverse == true
  /// and push_weights == true, then we push the weights in the reversed FSTs we create for each
  /// phone HMM.  This is only safe if the HMMs are probabilistic (i.e. not discriminatively
  bool push_weights;

  /// delta used if we do push_weights [only relevant if reverse == true
  /// and push_weights == true].
  BaseFloat push_delta;

  HTransducerConfig():
      trans_prob_scale(1.0),
      reverse(false),
      push_weights(true),
      push_delta(0.001)
  { }

  // Note-- this Register registers the easy-to-register options
  // but not the "sym_type" which is an enum and should be handled
  // separately in main().
  void Register (ParseOptions *po) {
    po->Register("transition-scale", &trans_prob_scale, "Scale of transition probs (relative to LM)");
    po->Register("reverse", &reverse, "Set true to build time-reversed FST.");
    po->Register("push-weights", &push_weights, "Push weights (only applicable if reverse == true)");
    po->Register("push-delta", &push_delta, "Delta used in pushing weights (only applicable if reverse && push-weights");
  }
};



/// Called by GetHTransducer() and probably will not need to be called directly;
/// it creates the FST corresponding to the phone.  Does not include self-loops;
/// you have to call AddSelfLoops() for that.  Result owned by caller.
/// Returns an acceptor (i.e. ilabels, olabels identical) with transition-ids
/// as the symbols.
/// For documentation in context, see \ref hmm_graph_get_hmm_as_fst
///   @param context_window  A vector representing the phonetic context; see
///            \ref tree_window "here" for explanation.
///   @param ctx_dep The object that contains the phonetic decision-tree
///   @param trans_model The transition-model object, which provides
///         the mappings to transition-ids and also the transition
///         probabilities.
///   @param config Configuration object, see \ref HTransducerConfig.

fst::VectorFst<fst::StdArc> *GetHmmAsFst(std::vector<int32> context_window,
                                         const ContextDependencyInterface &ctx_dep,
                                         const TransitionModel &trans_model,
                                         const HTransducerConfig &config);

/// Included mainly as a form of documentation, not used in any other code
/// currently.  Creates the FST with self-loops, and with fewer options.
fst::VectorFst<fst::StdArc>*
GetHmmAsFstSimple(std::vector<int32> context_window,
                  const ContextDependencyInterface &ctx_dep,
                  const TransitionModel &trans_model,
                  BaseFloat prob_scale);


/**
  * Returns the H tranducer; result owned by caller.
  * See \ref hmm_graph_get_h_transducer.  The H transducer has on the
  * input transition-ids, and also possibly some disambiguation symbols, which
  * will be put in disambig_syms.  The output side contains the identifiers that
  * are indexes into "ilabel_info" (these represent phones-in-context or
  * disambiguation symbols).  The ilabel_info vector allows GetHTransducer to map
  * from symbols to phones-in-context (i.e. phonetic context windows).  Any
  * singleton symbols in the ilabel_info vector which are not phones, will be
  * treated as disambiguation symbols.  [Not all recipes use these].  The output
  * "disambig_syms_left" will be set to a list of the disambiguation symbols on
  * the input of the transducer (i.e. same symbol type as whatever is on the
  * input of the transducer
  */
fst::VectorFst<fst::StdArc>*
GetHTransducer (const std::vector<std::vector<int32> > &ilabel_info,
                const ContextDependencyInterface &ctx_dep,
                const TransitionModel &trans_model,
                const HTransducerConfig &config,
                std::vector<int32> *disambig_syms_left);

/**
  * GetIlabelMapping produces a mapping that's similar to HTK's logical-to-physical
  * model mapping (i.e. the xwrd.clustered.mlist files).   It groups together
  * "logical HMMs" (i.e. in our world, phonetic context windows) that share the
  * same sequence of transition-ids.   This can be used in an
  * optional graph-creation step that produces a remapped form of CLG that can be
  * more productively determinized and minimized.  This is used in the command-line program
  * make-ilabel-transducer.cc.
  * @param ilabel_info_old [in] The original \ref tree_ilabel "ilabel_info" vector
  * @param ctx_dep [in] The tree
  * @param trans_model [in] The transition-model object
  * @param old2new_map [out] The output; this vector, which is of size equal to the
  *       number of new labels, is a mapping to the old labels such that we could
  *       create a vector ilabel_info_new such that
  *       ilabel_info_new[i] == ilabel_info_old[old2new_map[i]]
  */
void GetIlabelMapping (const std::vector<std::vector<int32> > &ilabel_info_old,
                       const ContextDependencyInterface &ctx_dep,
                       const TransitionModel &trans_model,
                       std::vector<int32> *old2new_map);



/**
  * For context, see \ref hmm_graph_add_self_loops.  Expands an FST that has been
  * built without self-loops, and adds the self-loops (it also needs to modify
  * the probability of the non-self-loop ones, as the graph without self-loops
  * was created in such a way that it was stochastic).  Note that the
  * disambig_syms will be empty in some recipes (e.g.  if you already removed
  * the disambiguation symbols).
  * @param trans_model [in] Transition model
  * @param disambig_syms [in] Sorted, uniq list of disambiguation symbols, required
  *       if the graph contains disambiguation symbols but only needed for sanity checks.
  * @param self_loop_scale [in] Transition-probability scale for self-loops; c.f.
  *                    \ref hmm_scale
  * @param reorder [in] If true, reorders the transitions (see \ref hmm_reorder).
  * @param  fst [in, out] The FST to be modified.
  */
void AddSelfLoops(const TransitionModel &trans_model,
                  const std::vector<int32> &disambig_syms,  // used as a check only.
                  BaseFloat self_loop_scale,
                  bool reorder,  // true->dan-style, false->lukas-style.
                  fst::VectorFst<fst::StdArc> *fst);

/**
  * Adds transition-probs, with the supplied
  * scales (see \ref hmm_scale), to the graph.
  * Useful if you want to create a graph without transition probs, then possibly
  * train the model (including the transition probs) but keep the graph fixed,
  * and add back in the transition probs.  It assumes the fst has transition-ids
  * on it.
  * @param trans_model [in] The transition modle
  * @param disambig_syms [in] A list of disambiguation symbols, required if the
  *                       graph has disambiguation symbols on its input but only
  *                       used for checks.
  * @param trans_prob_scale [in] A scale on transition-probabilities apart from
  *                      those involving self-loops; see \ref hmm_scale.
  * @param self_loop_scale [in] A scale on self-loop transition probabilities;
  *                      see \ref hmm_scale.
  * @param  fst [in, out] The FST to be modified.
  */
void AddTransitionProbs(const TransitionModel &trans_model,
                        const std::vector<int32> &disambig_syms,
                        BaseFloat trans_prob_scale,
                        BaseFloat self_loop_scale,
                        fst::VectorFst<fst::StdArc> *fst);


/// Returns a transducer from pdfs plus one (input) to  transition-ids (output).
/// Currenly of use only for testing.
fst::VectorFst<fst::StdArc>*
GetPdfToTransitionIdTransducer(const TransitionModel &trans_model);

/// Converts all transition-ids in the FST to pdfs plus one.
/// Placeholder: not implemented yet!
void ConvertTransitionIdsToPdfs(const TransitionModel &trans_model,
                                const std::vector<int32> &disambig_syms,
                                fst::VectorFst<fst::StdArc> *fst);

/// @} end "defgroup hmm_group_graph"

/// \addtogroup hmm_group
/// @{

/// SplitToPhones splits up the TransitionIds in "alignment" into their
/// individual phones (one vector per instance of a phone).  At output,
/// the sum of the sizes of the vectors in split_alignment will be the same
/// as the corresponding sum for "alignment".  The function returns
/// true on success.  If the alignment appears to be incomplete, e.g.
/// not ending at the end-state of a phone, it will still break it up into
/// phones but it will return false.  For more serious errors it will
/// die or throw an exception.
/// This function works out by itself whether the graph was created
/// with "reordering" (dan-style graph), and just does the right thing.

bool SplitToPhones(const TransitionModel &trans_model,
                   const std::vector<int32> &alignment,
                   std::vector<std::vector<int32> > *split_alignment);

/// ConvertAlignment converts an alignment that was created using one
/// model, to another model.  They must use a compatible topology (so we
/// know the state alignments of the new model).
/// It returns false if it could not be split to phones (probably
/// because the alignment was partial), but for other kinds of
/// error that are more likely a coding error, it will throw
/// an exception.
bool ConvertAlignment(const TransitionModel &old_trans_model,
                      const TransitionModel &new_trans_model,
                      const ContextDependencyInterface &new_ctx_dep,
                      const std::vector<int32> &old_alignment,
                      const std::vector<int32> *phone_map,  // may be NULL
                      std::vector<int32> *new_alignment);

} // end namespace kaldi

/// @} end "addtogroup hmm_group"

#endif
