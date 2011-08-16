// lat/lattice-utils.cc

// Copyright 2009-2011   Saarland University
// Author: Arnab Ghoshal

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
using std::pair;
#include <vector>
using std::vector;
#include <tr1/unordered_map>
using std::tr1::unordered_map;

#include "lat/lattice-utils.h"

namespace kaldi {

int32 LatticeStateTimes(const Lattice &lat, vector<int32> *times) {
  kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  int32 num_states = lat.NumStates();
  times->resize(num_states, -1);
  (*times)[0] = 0;
  for (int32 state = 0; state < num_states; ++state) {
    int32 cur_time = (*times)[state];
    for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
        aiter.Next()) {
      const LatticeArc& arc = aiter.Value();

      if (arc.ilabel != 0) {  // Non-epsilon input label on arc
        // next time instance
        if ((*times)[arc.nextstate] == -1) {
          (*times)[arc.nextstate] = cur_time + 1;
        } else {
          KALDI_ASSERT((*times)[arc.nextstate] == cur_time + 1);
        }
      } else {  // epsilon input label on arc
        // Same time instance
        if ((*times)[arc.nextstate] == -1)
          (*times)[arc.nextstate] = cur_time;
        else
          KALDI_ASSERT((*times)[arc.nextstate] == cur_time);
      }
    }
  }
  return (*std::max_element(times->begin(), times->end()));
}


// Helper functions for lattice forward-backward
static void ForwardNode(const Lattice &lat, int32 state,
                        vector<double> *state_alphas);
static void BackwardNode(const Lattice &lat, int32 state, int32 cur_time,
                         double tot_forward_prob,
                         const vector< vector<int32> > &active_states,
                         const vector<double> &state_alphas,
                         vector<double> *state_betas,
                         unordered_map<int32, double> *post);


BaseFloat LatticeForwardBackward(const Lattice &lat, Posterior *arc_post) {
  // Make sure the lattice is topologically sorted.
  kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);

  int32 num_states = lat.NumStates();
  vector<int32> state_times;
  int32 max_time = LatticeStateTimes(lat, &state_times);
  vector< vector<int32> > active_states(max_time + 1);
  // the +1 is needed since time is indexed from 0

  vector<double> state_alphas(num_states, kLogZeroDouble),
      state_betas(num_states, kLogZeroDouble);
  state_alphas[0] = 0.0;
  double tot_forward_prob = kLogZeroDouble;

  // Forward pass
  for (int32 state = 0; state < num_states; ++state) {
    int32 cur_time = state_times[state];
    active_states[cur_time].push_back(state);

    if (lat.Final(state) != LatticeWeight::Zero()) {  // Check if final state.
      state_betas[state] = 0.0;
      tot_forward_prob = LogAdd(tot_forward_prob, state_alphas[state]);
    } else {
      ForwardNode(lat, state, &state_alphas);
    }
  }

  // Backward pass and collect posteriors
  vector< unordered_map<int32, double> > tmp_arc_post(max_time);
  for (int32 state = num_states -1; state > 0; --state) {
    int32 cur_time = state_times[state];
    BackwardNode(lat, state, cur_time, tot_forward_prob, active_states,
                 state_alphas, &state_betas, &tmp_arc_post[cur_time - 1]);
  }
  double tot_backward_prob = state_betas[0];  // Initial state id == 0
  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-9)) {
    KALDI_ERR << "Total forward probability over lattice = " << tot_forward_prob
              << ", while total backward probability = " << tot_backward_prob;
  }

  // Output the computed posteriors
  arc_post->resize(max_time);
  for (int32 cur_time = 0; cur_time < max_time; ++cur_time) {
    unordered_map<int32, double>::const_iterator post_itr =
        tmp_arc_post[cur_time].begin();
    for (; post_itr != tmp_arc_post[cur_time].end(); ++post_itr) {
      (*arc_post)[cur_time].push_back(std::make_pair(post_itr->first,
                                                     post_itr->second));
    }
  }

  return tot_forward_prob;
}


// ----------------------- Helper function definitions -----------------------

// static
void ForwardNode(const Lattice &lat, int32 state,
                        vector<double> *state_alphas) {
  for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
      aiter.Next()) {
    const LatticeArc& arc = aiter.Value();
    double graph_score = arc.weight.Value1(),
        am_score = arc.weight.Value2(),
        arc_score = (*state_alphas)[state] - am_score - graph_score;
    (*state_alphas)[arc.nextstate] = LogAdd((*state_alphas)[arc.nextstate],
                                            arc_score);
  }
}

// static
void BackwardNode(const Lattice &lat, int32 state, int32 cur_time,
                         double tot_forward_prob,
                         const vector< vector<int32> > &active_states,
                         const vector<double> &state_alphas,
                         vector<double> *state_betas,
                         unordered_map<int32, double> *post) {
  // Epsilon arcs leading into the state
  for (vector<int32>::const_iterator st_it = active_states[cur_time].begin();
      st_it != active_states[cur_time].end(); ++st_it) {
    if ((*st_it) < state) {
      for (fst::ArcIterator<Lattice> aiter(lat, (*st_it)); !aiter.Done();
            aiter.Next()) {
        const LatticeArc& arc = aiter.Value();
        if (arc.nextstate == state) {
          KALDI_ASSERT(arc.ilabel == 0);
          double arc_score = (*state_betas)[state] - arc.weight.Value1()
              - arc.weight.Value2();
          (*state_betas)[(*st_it)] = LogAdd((*state_betas)[(*st_it)],
                                            arc_score);
        }
      }
    }
  }

  if (cur_time == 0) return;

  // Non-epsilon arcs leading into the state
  int32 prev_time = cur_time - 1;
  for (vector<int32>::const_iterator st_it = active_states[prev_time].begin();
      st_it != active_states[prev_time].end(); ++st_it) {
    for (fst::ArcIterator<Lattice> aiter(lat, (*st_it)); !aiter.Done();
        aiter.Next()) {
      const LatticeArc& arc = aiter.Value();
      if (arc.nextstate == state) {
        int32 key = arc.ilabel;
        KALDI_ASSERT(key != 0);
        double graph_score = arc.weight.Value1(),
            am_score = arc.weight.Value2(),
            arc_score = (*state_betas)[state] - graph_score - am_score;
        (*state_betas)[(*st_it)] = LogAdd((*state_betas)[(*st_it)],
                                          arc_score);
        BaseFloat gamma = std::exp(state_alphas[(*st_it)] - graph_score -
                                   am_score + (*state_betas)[state] -
                                   tot_forward_prob);
        unordered_map<int32, double>::iterator find_iter = post->find(key);
        if (find_iter == post->end()) {  // New label found at prev_time
          (*post)[key] = gamma;
        } else {  // Arc label already seen at this time
          (*post)[key] += gamma;
        }
      }
    }
  }
}


}  // namespace kaldi
