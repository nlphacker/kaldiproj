// decoder/faster-decoder.h

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

#ifndef KALDI_DECODER_FASTER_DECODER_H_
#define KALDI_DECODER_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "util/parse-options.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

namespace kaldi {

struct FasterDecoderOptions {
  BaseFloat beam;
  int32 max_active;
  BaseFloat beam_delta;
  BaseFloat hash_ratio;
  FasterDecoderOptions(): beam(16.0),
                          max_active(std::numeric_limits<int32>::max()),
                          beam_delta(0.5), hash_ratio(2.0) { }
  void Register(ParseOptions *po, bool full) {  /// if "full", use obscure
    /// options too.
    /// Depends on program.
    po->Register("beam", &beam, "Decoder beam");
    po->Register("max-active", &max_active, "Decoder max active states.");
    if (full) {
      po->Register("beam-delta", &beam_delta,
                   "Increment used in decoder [obscure setting]");
      po->Register("hash-ratio", &hash_ratio,
                   "Setting used in decoder to control hash behavior");
    }
  }
};

class FasterDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  // instantiate this class onece for each thing you have to decode.
  FasterDecoder(const fst::Fst<fst::StdArc> &fst,
                FasterDecoderOptions opts): fst_(fst), opts_(opts) {
    assert(opts_.hash_ratio >= 1.0);  // less doesn't make much sense.
    assert(opts_.max_active > 1);
    toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
  }

  void SetOptions(const FasterDecoderOptions &opts) { opts_ = opts; }

  ~FasterDecoder() {
    ClearToks(toks_.Clear());
  }

  void Decode(DecodableInterface *decodable) {
    // clean up from last time:
    ClearToks(toks_.Clear());
    StateId start_state = fst_.Start();
    assert(start_state != fst::kNoStateId);
    Arc dummy_arc(0, 0, Weight::One(), start_state);
    toks_.Insert(start_state, new Token(dummy_arc, NULL));
    ProcessNonemitting(std::numeric_limits<float>::max());
    for (int32 frame = 0; !decodable->IsLastFrame(frame-1); frame++) {
      BaseFloat adaptive_beam = ProcessEmitting(decodable, frame);
      ProcessNonemitting(adaptive_beam);
    }
  }

  bool ReachedFinal() {
    Weight best_weight = Weight::Zero();
    for (Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      Weight this_weight = Times(e->val->weight, fst_.Final(e->key));
      if (this_weight != Weight::Zero())
        return true;
    }
    return false;
  }

  bool GetBestPath(fst::MutableFst<LatticeArc> *fst_out) {
    // GetBestPath gets the decoding output.  If is_final == true, it limits itself
    // to final states; otherwise it gets the most likely token not taking into
    // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
    // nothing was available.  It returns true if it got output (thus, fst_out
    // will be nonempty).
    fst_out->DeleteStates();
    Token *best_tok = NULL;
    bool is_final = ReachedFinal();
    if (!is_final) {
      for (Elem *e = toks_.GetList(); e != NULL; e = e->tail)
        if (best_tok == NULL || *best_tok < *(e->val) )
          best_tok = e->val;
    } else {
      Weight best_weight = Weight::Zero();
      for (Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
        Weight this_weight = Times(e->val->weight, fst_.Final(e->key));
        if (this_weight != Weight::Zero() &&
           this_weight.Value() < best_weight.Value()) {
          best_weight = this_weight;
          best_tok = e->val;
        }
      }
    }
    if (best_tok == NULL) return false;  // No output.

    std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.
    for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
      BaseFloat amscore = tok->weight_a.Value(),
                lmscore = tok->arc_.weight.Value() - amscore;
      LatticeArc l_arc(tok->arc_.ilabel,
    	               tok->arc_.olabel,
    	               LatticeWeight(lmscore, amscore),
    	               tok->arc_.nextstate);
      arcs_reverse.push_back(l_arc);
    }
    assert(arcs_reverse.back().nextstate == fst_.Start());
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

    StateId cur_state = fst_out->AddState();
    fst_out->SetStart(cur_state);
    for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
      LatticeArc arc = arcs_reverse[i];
      arc.nextstate = fst_out->AddState();
      fst_out->AddArc(cur_state, arc);
      cur_state = arc.nextstate;
    }
    if (is_final) {
      fst_out->SetFinal(cur_state,
          LatticeWeight(fst_.Final(best_tok->arc_.nextstate).Value(), 0.0));
    } else {
      fst_out->SetFinal(cur_state, LatticeWeight::One());
    }
    KALDI_LOG << "best path, final:" << fst_.Final(best_tok->arc_.nextstate);
    //fst::FstPrinter<LatticeArc> fstprinter(*fst_out, NULL, NULL, NULL, false, true);
    //fstprinter.Print(&std::cout, "standard output");
    RemoveEpsLocal(fst_out);
    return true;
  }

 private:

  class Token {
   public:
    Arc arc_;
    Token *prev_;
    int32 ref_count_;
    Weight weight;
    Weight weight_a;
    inline Token(Arc &arc, Token *prev): arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        weight = Times(prev->weight, arc.weight);
      } else {
        weight = arc.weight;
      }
      weight_a = Weight::One();
    }
    inline bool operator < (const Token &other) {
      return weight.Value() > other.weight.Value();
      // This makes sense for log + tropical semiring.
    }

    inline ~Token() {
      assert(ref_count_ == 1);
      if (prev_ != NULL) TokenDelete(prev_);
    }
    inline static void TokenDelete(Token *tok) {
      if (tok->ref_count_ == 1) { 
        delete tok;
      } else {
        tok->ref_count_--;
      }
    }
  };
  typedef HashList<StateId, Token*>::Elem Elem;


  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count,
                      BaseFloat *adaptive_beam, Elem **best_elem) {
    BaseFloat best_weight = 1.0e+10;  // positive == high cost == bad.
    size_t count = 0;
    if (opts_.max_active == std::numeric_limits<int32>::max()) {
      for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
        BaseFloat w = static_cast<BaseFloat>(e->val->weight.Value());
        if (w < best_weight) {
          best_weight = w;
          if (best_elem) *best_elem = e;
        }
      }
      if (tok_count != NULL) *tok_count = count;
      if (adaptive_beam != NULL) *adaptive_beam = opts_.beam;
      return best_weight + opts_.beam;
    } else {
      tmp_array_.clear();
      for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
        BaseFloat w = e->val->weight.Value();
        tmp_array_.push_back(w);
        if (w < best_weight) {
          best_weight = w;
          if (best_elem) *best_elem = e;
        }
      }
      if (tok_count != NULL) *tok_count = count;
      if (tmp_array_.size() <= static_cast<size_t>(opts_.max_active)) {
        if (adaptive_beam) *adaptive_beam = opts_.beam;
        return best_weight + opts_.beam;
      } else {
        // the lowest elements (lowest costs, highest likes)
        // will be put in the left part of tmp_array.
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin()+opts_.max_active,
                         tmp_array_.end());
        // return the tighter of the two beams.
        BaseFloat ans = std::min(best_weight + opts_.beam,
                                 *(tmp_array_.begin()+opts_.max_active));
        if (adaptive_beam)
          *adaptive_beam = std::min(opts_.beam,
                                    ans - best_weight + opts_.beam_delta);
        return ans;
      }
    }
  }

  void PossiblyResizeHash(size_t num_toks) {
    size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                        * opts_.hash_ratio);
    if (new_sz > toks_.Size()) {
      toks_.SetSize(new_sz);
    }
  }

  // ProcessEmitting returns the likelihood cutoff used.
  BaseFloat ProcessEmitting(DecodableInterface *decodable, int frame) {
    Elem *last_toks = toks_.Clear();
    size_t tok_cnt;
    BaseFloat adaptive_beam;
    Elem *best_elem = NULL;
    BaseFloat weight_cutoff = GetCutoff(last_toks, &tok_cnt,
                                        &adaptive_beam, &best_elem);
    PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.

    // This is the cutoff we use after adding in the log-likes (i.e.
    // for the next frame).  This is a bound on the cutoff we will use
    // on the next frame.
    BaseFloat next_weight_cutoff = 1.0e+10;

    // First process the best token to get a hopefully
    // reasonably tight bound on the next cutoff.
    if (best_elem) {
      StateId state = best_elem->key;
      Token *tok = best_elem->val;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
          !aiter.Done();
          aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          arc.weight = Times(arc.weight,
                             Weight(- decodable->LogLikelihood(frame, arc.ilabel)));
          BaseFloat new_weight = arc.weight.Value() + tok->weight.Value();
          if (new_weight + adaptive_beam < next_weight_cutoff)
            next_weight_cutoff = new_weight + adaptive_beam;
        }
      }
    }

    // int32 n = 0, np = 0;

    // the tokens are now owned here, in last_toks, and the hash is empty.
    // 'owned' is a complex thing here; the point is we need to call DeleteElem
    // on each elem 'e' to let toks_ know we're done with them.
    for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {  // loop this way
      // n++;
      // because we delete "e" as we go.
      StateId state = e->key;
      Token *tok = e->val;
      if (tok->weight.Value() < weight_cutoff) {  // not pruned.
        // np++;
        assert(state == tok->arc_.nextstate);
        for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
            !aiter.Done();
            aiter.Next()) {
          Arc arc = aiter.Value();
          if (arc.ilabel != 0) {  // propagate..
            Weight amscore(- decodable->LogLikelihood(frame, arc.ilabel));
            arc.weight = Times(arc.weight, amscore);
            BaseFloat new_weight = arc.weight.Value() + tok->weight.Value();
            if (new_weight < next_weight_cutoff) {  // not pruned..
              Token *new_tok = new Token(arc, tok);
              new_tok->weight_a = amscore;
              Elem *e_found = toks_.Find(arc.nextstate);
              if (e_found == NULL) {
                toks_.Insert(arc.nextstate, new_tok);
              } else {
                if ( *(e_found->val) < *new_tok ) {
                  Token::TokenDelete(e_found->val);
                  e_found->val = new_tok;
                } else {
                  Token::TokenDelete(new_tok);
                }
              }
            }
          }
        }
      }
      e_tail = e->tail;
      Token::TokenDelete(e->val);
      toks_.Delete(e);
    }
    //  std::cerr << n << ', ' << np << ', ' <<adaptive_beam<<' ';
    return adaptive_beam;
  }

  void ProcessNonemitting(BaseFloat adaptive_beam) {
    // Processes nonemitting arcs for one frame.  Propagates within
    // cur_toks_.
    assert(queue_.empty());
    float best_weight = 1.0e+10;
    for (Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
      queue_.push_back(e->key);
      best_weight = std::min(best_weight, e->val->weight.Value());
    }
    BaseFloat cutoff = best_weight + adaptive_beam;

    while (!queue_.empty()) {
      StateId state = queue_.back();
      queue_.pop_back();
      Token *tok = toks_.Find(state)->val;  // would segfault if state not
      // in toks_ but this can't happen.
      if (tok->weight.Value() > cutoff) { // Don't bother processing successors.
        continue;
      }
      assert(tok != NULL && state == tok->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
          !aiter.Done();
          aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel == 0) {  // propagate nonemitting only...
          Token *new_tok = new Token(arc, tok);
          if (new_tok->weight.Value() > cutoff) {  // prune
            Token::TokenDelete(new_tok);
          } else {
            Elem *e_found = toks_.Find(arc.nextstate);
            if (e_found == NULL) {
              toks_.Insert(arc.nextstate, new_tok);
              queue_.push_back(arc.nextstate);
            } else {
              if ( *(e_found->val) < *new_tok ) {
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
                queue_.push_back(arc.nextstate);
              } else {
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }
  }

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  HashList<StateId, Token*> toks_;
  const fst::Fst<fst::StdArc> &fst_;
  FasterDecoderOptions opts_;
  std::vector<StateId> queue_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.


  // It might seem unclear why we call ClearToks(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void ClearToks(Elem *list) {
    for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
      Token::TokenDelete(e->val);
      e_tail = e->tail;
      toks_.Delete(e);
    }
  }

};


} // end namespace kaldi.


#endif
