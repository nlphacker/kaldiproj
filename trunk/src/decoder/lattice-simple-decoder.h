// decoder/lattice-simple-decoder.h

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

#ifndef KALDI_DECODER_LATTICE_SIMPLE_DECODER_H_
#define KALDI_DECODER_LATTICE_SIMPLE_DECODER_H_


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

#include <algorithm>
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;

namespace kaldi {

struct LatticeSimpleDecoderConfig {
  BaseFloat beam;
  BaseFloat lattice_beam;
  int32 prune_interval;
  bool determinize_lattice;
  bool debug_latgen;
  LatticeSimpleDecoderConfig(): beam(16.0),
                                lattice_beam(10.0),
                                prune_interval(25),
                                determinize_lattice(true),
                                debug_latgen(false) { }
  void Register(ParseOptions *po) {
    po->Register("beam", &beam, "Decoding beam.");
    po->Register("lattice-beam", &lattice_beam, "Lattice generation beam");
    po->Register("prune-interval", &prune_interval, "Interval (in frames) at which to prune tokens");
    po->Register("determinize-lattice", &determinize_lattice, "If true, determinize the lattice (in a special sense, keeping only best pdf-sequence for each word-sequence).");
    po->Register("debug-latgen", &debug_latgen, "If true, check memory consistency of lattice generation (very slow).");
  }
};


/** Simplest possible decoder, included largely for didactic purposes and as a
    means to debug more highly optimized decoders.  See \ref decoders_simple
    for more information.
 */
class LatticeSimpleDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  // instantiate this class onece for each thing you have to decode.
  LatticeSimpleDecoder(const fst::Fst<fst::StdArc> &fst,
                       const LatticeSimpleDecoderConfig &config):
      fst_(fst), config_(config), num_toks_(0) { }
  
  ~LatticeSimpleDecoder() {
    ClearActiveTokens();
  }

  bool Decode(DecodableInterface *decodable) {
    // clean up from last time:
    cur_toks_.clear();
    prev_toks_.clear();
    ClearActiveTokens();
    warned_ = false;
    final_active_ = false;
    final_costs_.clear();
    num_toks_ = 0;
    StateId start_state = fst_.Start();
    KALDI_ASSERT(start_state != fst::kNoStateId);
    active_toks_.resize(1);
    active_toks_[0].tok_head = active_toks_[0].tok_tail =
        cur_toks_[start_state] = new Token(0.0, 0.0, NULL, NULL, NULL);
    num_toks_++;
    ProcessNonemitting(0);
    // We use 1-based indexing for frames in this decoder (if you view it in
    // terms of features), but note that the decodable object uses zero-based
    // numbering, which we have to correct for when we call it.
    CheckLists();
    for (int32 frame = 1; !decodable->IsLastFrame(frame-2); frame++) {
      active_toks_.resize(frame+1);
      prev_toks_.clear();
      cur_toks_.swap(prev_toks_);
      CheckLists();
      ProcessEmitting(decodable, frame);
      CheckLists();
      // Important to call PruneCurrentTokens before ProcessNonemitting, or
      // we would get dangling forward pointers.  Anyway, ProcessNonemitting uses
      // the beam.
      CheckLists();
      PruneCurrentTokens(config_.beam, &cur_toks_);
      CheckLists();
      ProcessNonemitting(frame);
      CheckLists();
      if (decodable->IsLastFrame(frame-1))
        PruneActiveTokensFinal(frame);
      else if (frame % config_.prune_interval == 0)
        PruneActiveTokens(frame, config_.lattice_beam * 0.1); // use larger delta.        
    }
    // Returns true if we have any kind of traceback available (not necessarily
    // to the end state; query ReachedFinal() for that).
    return !final_costs_.empty();
  }

  /// says whether a final-state was active on the last frame.  If it was not, the
  /// lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() { return final_active_; }


  // Outputs an FST corresponding to the single best path
  // through the lattice.
  bool GetTraceback(fst::MutableFst<LatticeArc> *ofst) const {
    fst::VectorFst<LatticeArc> fst;
    if (!GetRawLattice(&fst)) return false;
    // std::cout << "Raw lattice is:\n";
    // fst::FstPrinter<LatticeArc> fstprinter(fst, NULL, NULL, NULL, false, true);
    // fstprinter.Print(&std::cout, "standard output");
    ShortestPath(fst, ofst);
    return true;
  }

  // Outputs an FST corresponding to the raw, state-level
  // tracebacks.
  bool GetRawLattice(fst::MutableFst<LatticeArc> *ofst) const {
    typedef LatticeArc Arc;
    typedef Arc::StateId StateId;
    typedef Arc::Weight Weight;
    typedef Arc::Label Label;
    ofst->DeleteStates();
    // num-frames plus one (since frames are one-based, and we have
    // an extra frame for the start-state).
    int32 num_frames = active_toks_.size() - 1;
    KALDI_ASSERT(num_frames > 0);
    unordered_map<Token*, StateId> tok_map(num_toks_/2 + 3);
    // First create all states.
    for (int32 f = 0; f <= num_frames; f++) {
      if (active_toks_[f].tok_head == NULL) {
        KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                   << ": not producing lattice.\n";
        return false;
      }
      for (Token *tok = active_toks_[f].tok_head;
          tok != NULL;
           tok = tok->next)
        tok_map[tok] = ofst->AddState();
    }
    ofst->SetStart(0);
    StateId cur_state = 0; // we rely on the fact that we numbered these
    // consecutively (AddState() returns the numbers in order..)
    for (int32 f = 0; f <= num_frames; f++) {
      for (Token *tok = active_toks_[f].tok_head;
           tok != NULL;
           tok = tok->next, cur_state++) {
        for (ForwardLink *l = tok->links;
             l != NULL;
             l = l->next) {
          unordered_map<Token*, StateId>::const_iterator iter =
              tok_map.find(l->next_tok);
          StateId nextstate = iter->second;
          KALDI_ASSERT(iter != tok_map.end());
          Arc arc(l->ilabel, l->olabel,
                  Weight(l->graph_cost, l->acoustic_cost),
                  nextstate);
          ofst->AddArc(cur_state, arc);
        }
        if (f == num_frames) {
          std::map<Token*, BaseFloat>::const_iterator iter =
              final_costs_.find(tok);
          if (iter != final_costs_.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        }
      }
    }
    KALDI_ASSERT(cur_state == ofst->NumStates());
    return (cur_state != 0);
  }

  // Outputs an FST corresponding to the lattice-determinized
  // lattice (one path per word sequence).
  bool GetLattice(fst::MutableFst<CompactLatticeArc> *ofst) const {
    fst::VectorFst<LatticeArc> raw_fst;
    if(!GetRawLattice(&raw_fst)) return false;
    Invert(&raw_fst); // make it so word labels are on the input.
    DeterminizeLattice(raw_fst, ofst);
    return true;
  }
  
  
  /*
  bool GetOutput(bool is_final, fst::MutableFst<fst::StdArc> *fst_out) {  
    // GetOutput gets the decoding output.  If is_final == true, it limits itself to final states;
    // otherwise it gets the most likely token not taking into account final-probs.
    // fst_out will be empty (Start() == kNoStateId) if nothing was available.
    // It returns true if it got output (thus, fst_out will be nonempty).
    fst_out->DeleteStates();
    Token *best_tok = NULL;
    if (!is_final) {
      for (unordered_map<StateId, Token*>::iterator iter = cur_toks_.begin();
          iter != cur_toks_.end();
          ++iter)
        if (best_tok == NULL || *best_tok < *(iter->second) )
          best_tok = iter->second;
    } else {
      Weight best_weight = Weight::Zero();
      for (unordered_map<StateId, Token*>::iterator iter = cur_toks_.begin();
          iter != cur_toks_.end();
          ++iter) {
        Weight this_weight = Times(iter->second->arc_.weight, fst_.Final(iter->first));
        if (this_weight != Weight::Zero() &&
           this_weight.Value() < best_weight.Value()) {
          best_weight = this_weight;
          best_tok = iter->second;
        }
      }
    }
    if (best_tok == NULL) return false;  // No output.

    std::vector<Arc> arcs_reverse;  // arcs in reverse order.
    for (Token *tok = best_tok; tok != NULL; tok = tok->prev_)
      arcs_reverse.push_back(tok->arc_);
    KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

    StateId cur_state = fst_out->AddState();
    fst_out->SetStart(cur_state);
    for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
      Arc arc = arcs_reverse[i];
      arc.nextstate = fst_out->AddState();
      fst_out->AddArc(cur_state, arc);
      cur_state = arc.nextstate;
    }
    if (is_final)
      fst_out->SetFinal(cur_state, fst_.Final(best_tok->arc_.nextstate));
    else
      fst_out->SetFinal(cur_state, Weight::One());
    RemoveEpsLocal(fst_out);
    return true;
    }
  */

 private:
  struct Token;
  // ForwardLinks are the links from a token to a token on the next frame.
  // or sometimes on the current frame (for input-epsilon links).
  struct ForwardLink {
    Token *next_tok; // the next token [or NULL if represents final-state]
    Label ilabel; // ilabel on link.
    Label olabel; // olabel on link.
    BaseFloat graph_cost; // graph cost of traversing link (contains LM, etc.)
    BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
    ForwardLink *next; // next in singly-linked list of forward links from a
                       // token.
    ForwardLink(Token *next_tok, Label ilabel, Label olabel,
                BaseFloat graph_cost, BaseFloat acoustic_cost, 
                ForwardLink *next):
        next_tok(next_tok), ilabel(ilabel), olabel(olabel),
        graph_cost(graph_cost), acoustic_cost(acoustic_cost), 
        next(next) { }
  };  
  
  // Token is what's resident in a particular state at a particular time.
  // In this decoder a Token actually contains *forward* links.
  // When first created, a Token just has the (total) cost.    We add forward
  // links to it when we process the next frame.
  struct Token {
    BaseFloat tot_cost; // would equal weight.Value()... cost up to this point.
    BaseFloat extra_cost; // >= 0.  After calling PruneForwardLinks, this equals
    // the minimum difference between the cost of the best path, and the cost of
    // this is on, and the cost of the absolute best path, under the assumption
    // that any of the currently active states at the decoding front may
    // eventually succeed (e.g. if you were to take the currently active states
    // one by one and compute this difference, and then take the minimum).
    
    ForwardLink *links; // Head of singly linked list of ForwardLinks
    
    Token *next; // "next" and "prev" are links in a per-frame doubly linked list
    Token *prev; // of Tokens, for the whole utterance.
    Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links, Token *next,
          Token *prev): tot_cost(tot_cost), extra_cost(extra_cost), links(links),
                        next(next), prev(prev) { }
    Token() {}
    void DeleteForwardLinks() {
      ForwardLink *l = links, *m; 
      while (l != NULL) {
        m = l->next;
        delete l;
        l = m;
      }
      links = NULL;
    }
  };
  
  // head and tail of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *tok_head;
    Token *tok_tail;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    TokenList(): tok_head(NULL), tok_tail(NULL), must_prune_forward_links(true),
                 must_prune_tokens(true) { }
  };
  

  // AddToken inserts a new, empty token (i.e. with no forward links) for the
  // given frame.  [note: it's inserted if necessary into cur_toks_ and also into
  // the doubly linked list of tokens active on this frame (whose head and tail is
  // at active_toks_[frame]).
  //
  // If "emitting" is false, then we're a bit careful about the order (since we
  // need to maintain the tokens in topological order); in this case, it will
  // move any existing token to the end of the doubly linked list of tokens for
  // the current frame.
  //
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  inline Token *AddToken(StateId state, int32 frame, BaseFloat tot_cost,
                         bool emitting, bool *changed) {
    KALDI_ASSERT(frame < active_toks_.size());
    Token *&tok_head = active_toks_[frame].tok_head,
        *&tok_tail = active_toks_[frame].tok_tail;
    
    unordered_map<StateId, Token*>::iterator find_iter = cur_toks_.find(state);
    if (find_iter == cur_toks_.end()) { // no such token presently.
      // Create one.
      Token *new_tok = new Token;
      num_toks_++;
      new_tok->tot_cost = tot_cost;
      new_tok->extra_cost = 0; // tokens on the currently final frame have zero extra_cost
                          // as any of them could end up
          // on the winning path.
      new_tok->links = NULL; // forward links: will be populated later.
      new_tok->next = NULL; // since new_tok will be the tail of the list.
      new_tok->prev = tok_tail;
      if (tok_tail) tok_tail->next = new_tok;
      else tok_head = new_tok;
      tok_tail = new_tok;
      cur_toks_[state] = new_tok;
      if (changed) *changed = true;
      return new_tok;
    } else {
      Token *tok = find_iter->second; // There is an existing Token for this state.
      if (tok->tot_cost > tot_cost) {
        tok->tot_cost = tot_cost;
        if (changed) *changed = true;
      } else {
        if (changed) *changed = false;
      }
      if (!emitting && tok != tok_tail) {
        // Excise tok from list; put at tail of list.  This is necessary to
        // maintain nonemitting tokens in topological order, which is necessary
        // for the token-pruning algorithm (PruneActiveTokens).
        tok->next->prev = tok->prev;
        // note: tok->next != NULL since tok != tok_tail
        if (tok->prev != NULL) tok->prev->next = tok->next;
        else tok_head = tok->next;
        // At this point the token is excised; now we put it at tail
        // of list.
        tok->prev = tok_tail;
        tok_tail->next = tok;
        tok->next = NULL;
        tok_tail = tok;
      }
      return tok;
    }
  }
  
  // Deletes a token, and any forward pointers it has.
  // Excise from doubly linked list of tokens.
  void DeleteToken(int32 frame, Token *tok) {
    KALDI_ASSERT(frame < active_toks_.size());
    Token *&tok_head = active_toks_[frame].tok_head,
        *&tok_tail = active_toks_[frame].tok_tail;
    tok->DeleteForwardLinks();
    if (tok->next != NULL) tok->next->prev = tok->prev;
    else { KALDI_ASSERT(tok_tail == tok);  tok_tail = tok->prev; }
    if (tok->prev != NULL) tok->prev->next = tok->next;
    else { KALDI_ASSERT(tok_head == tok); tok_head = tok->next; }
    delete tok;
    num_toks_--;
  }

  void CheckLists() {
    if(!config_.debug_latgen) return;
    for(size_t frame = 0; frame < active_toks_.size(); frame++) {
      for(Token *tok = active_toks_[frame].tok_head;
          tok != NULL;
          tok = tok->next) {
        if (tok->prev){ KALDI_ASSERT(tok->prev->next == tok); }
        else { KALDI_ASSERT(tok == active_toks_[frame].tok_head); }
        if (tok->next){ KALDI_ASSERT(tok->next->prev == tok); }
        else { KALDI_ASSERT(tok == active_toks_[frame].tok_tail); }
        for(ForwardLink *link = tok->links;
            link != NULL;
            link = link->next) {
          Token *next_tok = link->next_tok;
          if (next_tok->prev){ KALDI_ASSERT(next_tok->prev->next == next_tok); }
          else {
            size_t next_frame = frame + (link->ilabel == 0 ? 0 : 1);
            if (next_tok->prev){ KALDI_ASSERT(next_tok->prev->next == next_tok); }
            else{ KALDI_ASSERT(next_tok == active_toks_[next_frame].tok_head); }
            if (next_tok->next){ KALDI_ASSERT(next_tok->next->prev == next_tok); }
            else { KALDI_ASSERT(next_tok == active_toks_[next_frame].tok_tail); }
          }                              
        }
      }
    }
  }

  void PruneForwardLinks(int32 frame, bool *extra_costs_changed, bool *links_pruned,
                         BaseFloat delta) { // delta is the amount by which the extra_costs must
    // change before it sets "extra_costs_changed" to true.  If delta is larger,
    // we'll tend to go back less far toward the beginning of the file.
    *extra_costs_changed = false;
    *links_pruned = false;
    KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
    if (active_toks_[frame].tok_head == NULL ) { // empty list; this should
      // not happen.
      if (!warned_) {
        KALDI_WARN << "No tokens alive [doing pruning].. warning first "
            "time only for each utterance\n";
        warned_ = true;
      }
    }
    // Go through tokens on this frame in reverse order (required to correctly
    // handle epsilon links; they are in topological order).
    for (Token *tok = active_toks_[frame].tok_tail;
         tok != NULL;
         tok = tok->prev) {
      ForwardLink *link, *prev_link=NULL;
      // will recompute tok_extra_cost.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
          *links_pruned = true;
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;
          link = link->next;
        }
      }
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        *extra_costs_changed = true;
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  }

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses the final-probs
  // for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal(int32 frame) {
    KALDI_ASSERT(static_cast<size_t>(frame+1) == active_toks_.size());
    if (active_toks_[frame].tok_head == NULL ) // empty list; this should
      // not happen.
      KALDI_WARN << "No tokens alive at end of file\n";

    // First go through, working out the best token (do it in parallel
    // including final-probs and not including final-probs; we'll take
    // the one with final-probs if it's valid).
    const BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
    BaseFloat best_cost_final = infinity,
        best_cost_nofinal = infinity;
    unordered_map<Token*, StateId> tok_to_state_map;

    unordered_map<StateId, Token*>::iterator iter;
    for(iter = cur_toks_.begin(); iter != cur_toks_.end(); ++iter) {
      StateId state = iter->first;
      Token *tok = iter->second;
      tok_to_state_map[tok] = state;
      best_cost_final = std::min(best_cost_final,
                                 tok->tot_cost + fst_.Final(state).Value());
      best_cost_nofinal = std::min(best_cost_nofinal, tok->tot_cost);
    }
    final_active_ = (best_cost_final != infinity);

    // Go through tokens on this frame in reverse order (required to correctly
    // handle epsilon links; they are in topological order).
    for (Token *tok = active_toks_[frame].tok_tail;
         tok != NULL;
         tok = tok->prev) {
      ForwardLink *link, *prev_link=NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this token,
      // and the best such (score+final_prob).
      BaseFloat tok_extra_cost;
      if (final_active_) {
        BaseFloat final_cost = fst_.Final(tok_to_state_map[tok]).Value();
        tok_extra_cost = (tok->tot_cost + final_cost) - best_cost_final;
      } else 
        tok_extra_cost = tok->tot_cost - best_cost_nofinal;
      
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in non-final case because then, this case showed up as
      // having no forward links.  Here, the tok_extra_cost has an extra component
      // relating to the final-prob.
      if(tok_extra_cost > config_.lattice_beam)
        tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
      if (tok_extra_cost != std::numeric_limits<BaseFloat>::infinity()) {
        // If the token was not pruned away, 
        if(final_active_) {
          BaseFloat final_cost = fst_.Final(tok_to_state_map[tok]).Value();
          if (final_cost != std::numeric_limits<BaseFloat>::infinity())
            final_costs_[tok] = final_cost;
        } else {
          final_costs_[tok] = 0;
        }
      }
    }
  }

  
  // Prune away any tokens on this frame that have no forward links. [we don't do
  // this in PruneForwardLinks because it would give us a problem with dangling
  // pointers].
  void PruneTokensForFrame(int32 frame) {
    KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
    Token *&tok_head = active_toks_[frame].tok_head,
        *&tok_tail = active_toks_[frame].tok_tail;
    if (tok_head == NULL)
      KALDI_WARN << "No tokens alive [doing pruning]\n";
    Token *tok, *next_tok;
    for (tok = tok_head;
         tok != NULL;
         tok = next_tok) {
      next_tok = tok->next;
      if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
        // Next token is unreachable from end of graph; excise tok from list
        // and delete tok.
        if (tok->prev) tok->prev->next = tok->next;
        else tok_head = tok->next;
        if (tok->next) tok->next->prev = tok->prev;
        else tok_tail = tok->prev;
        delete tok;
        num_toks_--;
      }
    }
  }
  
  // Go backwards through still-alive tokens, pruning them.  note: cur_frame is
  // where cur_toks_ are (so we do not want to mess with it because these tokens
  // don't yet have forward pointers), but we do all previous frames, unless we
  // know that we can safely ignore them becaus the frame after them was unchanged.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse less
  // far.
  void PruneActiveTokens(int32 cur_frame, BaseFloat delta) {
    CheckLists();
    int32 num_toks_begin = num_toks_;
    for (int32 frame = cur_frame-1; frame >= 0; frame--) {
      CheckLists();
      // Reason why we need to prune forward links in this situation:
      // (1) we have never pruned them
      // (2) we never pruned the forward links on the next frame, which
      //     
      if (active_toks_[frame].must_prune_forward_links) {
        bool extra_costs_changed, links_pruned;
        PruneForwardLinks(frame, &extra_costs_changed, &links_pruned, delta);
        if (extra_costs_changed && frame > 0)
          active_toks_[frame-1].must_prune_forward_links = true;
        if (links_pruned)
          active_toks_[frame].must_prune_tokens = true;
        active_toks_[frame].must_prune_forward_links = false;
      }
      CheckLists();
      if (frame+1 < cur_frame &&
         active_toks_[frame+1].must_prune_tokens) {
        PruneTokensForFrame(frame+1);
        active_toks_[frame+1].must_prune_tokens = false;
      }
    }
    KALDI_VLOG(1) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                  << " to " << num_toks_;
  }

  // Version of PruneActiveTokens that we call on the final
  // frame.  Takes into account the final-prob of tokens.
  // returns true if there were final states active (else it treats
  // all states as final while doing the pruning, and returns false--
  // this can be useful if you want partial lattice output, although
  // it can be dangerous, depending what you want the lattices for).
  // final_active_ is set intenally (by PruneForwardLinksFinal),
  // and final_probs_ (a hash) is also set by PruneForwardLinksFinal.
  void PruneActiveTokensFinal(int32 cur_frame) {
    CheckLists();
    int32 num_toks_begin = num_toks_;
    PruneForwardLinksFinal(cur_frame); 
    for (int32 frame = cur_frame-1; frame >= 0; frame--) {
      CheckLists();
      bool b1, b2; // values not used.
      BaseFloat dontcare = 0.0;
      PruneForwardLinks(frame, &b1, &b2, dontcare);
      CheckLists();
      PruneTokensForFrame(frame+1);
    }
    KALDI_VLOG(1) << "PruneActiveTokensFinal: pruned tokens from " << num_toks_begin
                  << " to " << num_toks_;
  }
    
  
  void ProcessEmitting(DecodableInterface *decodable, int32 frame) {
    // Processes emitting arcs for one frame.  Propagates from
    // prev_toks_ to cur_toks_.
    BaseFloat cutoff = std::numeric_limits<BaseFloat>::infinity();
    for (unordered_map<StateId, Token*>::iterator iter = prev_toks_.begin();
         iter != prev_toks_.end();
         ++iter) {
      StateId state = iter->first;
      Token *tok = iter->second;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
          !aiter.Done();
          aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost = -decodable->LogLikelihood(frame-1, arc.ilabel),
              graph_cost = arc.weight.Value(),
              cur_cost = tok->tot_cost,
              tot_cost = cur_cost + ac_cost + graph_cost;
          if (tot_cost > cutoff) continue;
          else if (tot_cost + config_.beam < cutoff)
            cutoff = tot_cost + config_.beam;
          // AddToken adds the next_tok to cur_toks_ (if not already present).
          Token *next_tok = AddToken(arc.nextstate, frame, tot_cost, true, NULL);
          
          // Add ForwardLink from tok to next_tok (put on head of list tok->links)
          tok->links = new ForwardLink(next_tok, arc.ilabel, arc.olabel, 
                                       graph_cost, ac_cost, tok->links);
        }
      }
    }
  }

  void ProcessNonemitting(int32 frame) { // note: "frame" is the same as emitting states
    // just processed.
    
    // Processes nonemitting arcs for one frame.  Propagates within
    // cur_toks_.  Note-- this queue structure is is not very optimal as
    // it may cause us to process states unnecessarily (e.g. more than once),
    // but in the baseline code, turning this vector into a set to fix this
    // problem did not improve overall speed.
    std::vector<StateId> queue_;
    float best_cost = std::numeric_limits<BaseFloat>::infinity();
    for (unordered_map<StateId, Token*>::iterator iter = cur_toks_.begin();
         iter != cur_toks_.end();
         ++iter) {
      queue_.push_back(iter->first);
      best_cost = std::min(best_cost, iter->second->tot_cost);
    }
    if (queue_.empty()) {
      if (!warned_) {
        KALDI_ERR << "Error in ProcessEmitting: no surviving tokens: frame is "
                  << frame;
        warned_ = true;
      }
    }
    BaseFloat cutoff = best_cost + config_.beam;
    
    while (!queue_.empty()) {
      StateId state = queue_.back();
      queue_.pop_back();
      Token *tok = cur_toks_[state];
      // If "tok" has any existing forward links, delete them,
      // because we're about to regenerate them.  This is a kind
      // of non-optimality (remember, this is the simple decoder),
      // but since most states are emitting it's not a huge issue.
      tok->DeleteForwardLinks();
      tok->links = NULL;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
          !aiter.Done();
          aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel == 0) {  // propagate nonemitting only...
          BaseFloat graph_cost = arc.weight.Value(),
              cur_cost = tok->tot_cost,
              tot_cost = cur_cost + graph_cost;
          if (tot_cost < cutoff) {
            bool changed;
            Token *new_tok = AddToken(arc.nextstate, frame, tot_cost,
                                      false, &changed);
            
            tok->links = new ForwardLink(new_tok, 0, arc.olabel,
                                         graph_cost, 0, tok->links);
            
            // "changed" tells us whether the new token has a different
            // cost from before, or is new [if so, add into queue].
            if (changed)
              queue_.push_back(arc.nextstate);
          }
        }
      }
    }
  }

  unordered_map<StateId, Token*> cur_toks_;
  unordered_map<StateId, Token*> prev_toks_;
  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are tok_head, tok_tail, ever_pruned).  
  const fst::Fst<fst::StdArc> &fst_;
  LatticeSimpleDecoderConfig config_;
  int32 num_toks_; // current total #toks allocated...
  bool warned_;
  bool final_active_; // use this to say whether we found active final tokens
  // on the last frame.
  std::map<Token*, BaseFloat> final_costs_; // A cache of final-costs
  // of tokens on the last frame-- it's just convenient to storeit this way.
  
  void ClearActiveTokens() { // a cleanup routine, at utt end/begin
    for (size_t i = 0; i < active_toks_.size(); i++) {
      // Delete all tokens alive on this frame, and any forward
      // links they may have.
      Token *tok = active_toks_[i].tok_head;
      while (tok != NULL) {
        tok->DeleteForwardLinks();
        Token *next_tok = tok->next;
        delete tok;
        num_toks_--;
        tok = next_tok;
      }
    }
    active_toks_.clear();
    KALDI_ASSERT(num_toks_ == 0);
  }

  // PruneCurrentTokens deletes the tokens from the "toks" map, but not
  // from the active_toks_ list, which could cause dangling forward pointers
  // (will delete it during regular pruning operation).
  void PruneCurrentTokens(BaseFloat beam, unordered_map<StateId, Token*> *toks) {
    if (toks->empty()) {
      KALDI_VLOG(2) <<  "No tokens to prune.\n";
      return;
    }
    BaseFloat best_cost = 1.0e+10;  // positive == high cost == bad.
    for (unordered_map<StateId, Token*>::iterator iter = toks->begin();
        iter != toks->end(); ++iter) {
      best_cost =
          std::min(best_cost,
                   static_cast<BaseFloat>(iter->second->tot_cost));
    }
    std::vector<StateId> retained;
    BaseFloat cutoff = best_cost + beam;
    for (unordered_map<StateId, Token*>::iterator iter = toks->begin();
        iter != toks->end(); ++iter) {
      if (iter->second->tot_cost < cutoff)
        retained.push_back(iter->first);
    }
    unordered_map<StateId, Token*> tmp;
    for (size_t i = 0; i < retained.size(); i++) {
      tmp[retained[i]] = (*toks)[retained[i]];
    }
    KALDI_VLOG(2) <<  "Pruned to "<<(retained.size())<<" toks.\n";
    tmp.swap(*toks);
  }
};


} // end namespace kaldi.


#endif
