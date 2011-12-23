// fstext/lattice-utils.h

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


#ifndef KALDI_FSTEXT_LATTICE_UTILS_H_
#define KALDI_FSTEXT_LATTICE_UTILS_H_

#include "fst/fstlib.h"
#include "fstext/lattice-weight.h"
#include "fstext/factor.h"

namespace fst {

// The template ConvertLattice does conversions to and from
// LatticeWeight FSTs and CompactLatticeWeight FSTs, and
// between float and double, and to convert from LatticeWeight
// to StdWeight.  It's used in the I/O code for lattices,
// and for converting lattices to standard FSTs (e.g. for creating
// decoding graphs from lattices).


/**
   Convert lattice from a normal FST to a CompactLattice FST.
   This is a bit like converting to the Gallic semiring, except
   the semiring behaves in a different way (designed to take
   the best path).
   Note: the ilabels end up as the symbols on the arcs of the
   output acceptor, and the olabels go to the strings.  To make
   it the other way around (useful for the speech-recognition
   application), set invert=true [the default].
*/
template<class Weight, class Int>
void ConvertLattice(
    const ExpandedFst<ArcTpl<Weight> > &ifst,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight,Int> > > *ofst,
    bool invert = true);

/**
   Convert lattice from a CompactLattice FST to a normal FST.  This is a bit like
   converting from the Gallic semiring.  "ifst" must be an acceptor (i.e.,
   ilabels and olabels should be identical).  If invert=false, the labels on
   "ifst" become the ilabels on "ofst" and the strings in the weights of "ifst"
   becomes the olabels.  If invert=true [default], this is reversed (useful for speech
   recognition lattices; our standard non-compact format has the words on the output side to
   match HCLG).
*/
template<class Weight, class Int>
void ConvertLattice(
    const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<Weight,Int> > > &ifst,
    MutableFst<ArcTpl<Weight> > *ofst,
    bool invert = true);


/**
  Convert between CompactLattices of different floating point types...
  this works between any pair of weight types for which ConvertLatticeWeight
  is defined (c.f. lattice-weight.h)... 
 */
template<class WeightIn, class WeightOut>
void ConvertLattice(
    const ExpandedFst<ArcTpl<WeightIn> > &ifst,
    MutableFst<ArcTpl<WeightOut> > *ofst);


// Now define some ConvertLattice functions that require two phases of conversion (don't
// bother coding these separately as they will be used rarely.
template<class Int>
void ConvertLattice(const ExpandedFst<ArcTpl<LatticeWeightTpl<float> > > &ifst,
                    MutableFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<double>, Int> > > *ofst) {
  VectorFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, Int> > > fst;
  ConvertLattice(ifst, &fst);
  ConvertLattice(fst, ofst);
}

template<class Int>
void ConvertLattice(const ExpandedFst<ArcTpl<LatticeWeightTpl<double> > > &ifst,
                    MutableFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, Int> > > *ofst) {
  VectorFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<double>, Int> > > fst;
  ConvertLattice(ifst, &fst);
  ConvertLattice(fst, ofst);
}

template<class Int>
void ConvertLattice(const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<double>, Int> > > &ifst,
                    MutableFst<ArcTpl<LatticeWeightTpl<float> > > *ofst) {
  VectorFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, Int> > > fst;
  ConvertLattice(ifst, &fst);
  ConvertLattice(fst, ofst);
}

template<class Int>
void ConvertLattice(const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, Int> > > &ifst,
                    MutableFst<ArcTpl<LatticeWeightTpl<double> > > *ofst) {
  VectorFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<double>, Int> > > fst;
  ConvertLattice(ifst, &fst);
  ConvertLattice(fst, ofst);
}

template<class Weight, class Int>
void ConvertLattice(const ExpandedFst<ArcTpl<LatticeWeightTpl<Weight> > > &ifst,
                    MutableFst<ArcTpl<Weight> > *ofst,
                    bool invert = true) {
  VectorFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<Weight>, Int> > > fst;
  ConvertLattice(ifst, &fst);
  ConvertLattice(fst, ofst, invert);
}

/** Returns a default 2x2 matrix scaling factor for LatticeWeight */
inline vector<vector<double> > DefaultLatticeScale() {
  vector<vector<double> > ans(2);
  ans[0].resize(2, 0.0);
  ans[1].resize(2, 0.0);
  ans[0][0] = ans[1][1] = 1.0;
  return ans;
}

inline vector<vector<double> > AcousticLatticeScale(double acwt) {
  vector<vector<double> > ans(2);
  ans[0].resize(2, 0.0);
  ans[1].resize(2, 0.0);
  ans[0][0] = 1.0;
  ans[1][1] = acwt;
  return ans;
}

inline vector<vector<double> > GraphLatticeScale(double lmwt) {
  vector<vector<double> > ans(2);
  ans[0].resize(2, 0.0);
  ans[1].resize(2, 0.0);
  ans[0][0] = lmwt;
  ans[1][1] = 1.0;
  return ans;
}

inline vector<vector<double> > LatticeScale(double lmwt, double acwt) {
  vector<vector<double> > ans(2);
  ans[0].resize(2, 0.0);
  ans[1].resize(2, 0.0);
  ans[0][0] = lmwt;
  ans[1][1] = acwt;
  return ans;
}


/** Scales the pairs of weights in LatticeWeight or CompactLatticeWeight by
    viewing the pair (a, b) as a 2-vector and pre-multiplying by the 2x2 matrix
    in "scale".  E.g. typically scale would equal
     [ 1   0;
       0  acwt ]
    if we want to scale the acoustics by "acwt".
 */
template<class Weight, class ScaleFloat>
void ScaleLattice(
    const vector<vector<ScaleFloat> > &scale,
    MutableFst<ArcTpl<Weight> > *fst);

/// Removes state-level alignments (the strings that are
/// part of the weights).
template<class Weight, class Int>
void RemoveAlignmentsFromCompactLattice(
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, Int> > > *fst);

/// Returns true if lattice has alignments, i.e. it has
/// any nonempty strings inside its weights.
template<class Weight, class Int>
bool CompactLatticeHasAlignment(
    const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<Weight, Int> > > &fst);


/// Class StdToLatticeMapper maps a normal arc (StdArc)
/// to a LatticeArc by putting the StdArc weight as the first
/// element of the LatticeWeight.  Useful when doing LM
/// rescoring.

template<class Int>
class StdToLatticeMapper {
  typedef LatticeWeightTpl<Int> LatticeWeight;
  typedef ArcTpl<LatticeWeight> LatticeArc;
 public:
  LatticeArc operator()(const StdArc &arc) {
    return LatticeArc(arc.ilabel, arc.olabel,
                      LatticeWeight(arc.weight.Value(), 0.0),
                      arc.nextstate);
  }
  MapFinalAction FinalAction() { return MAP_NO_SUPERFINAL; }

  MapSymbolsAction InputSymbolsAction() { return MAP_COPY_SYMBOLS; }

  MapSymbolsAction OutputSymbolsAction() { return MAP_COPY_SYMBOLS; }

  // I believe all properties are preserved.
  uint64 Properties(uint64 props) { return props; }
};


/// Class LatticeToStdMapper maps a LatticeArc to a normal arc (StdArc)
/// by adding the elements of the LatticeArc weight.

template<class Int>
class LatticeToStdMapper {
  typedef LatticeWeightTpl<Int> LatticeWeight;
  typedef ArcTpl<LatticeWeight> LatticeArc;
 public:
  StdArc operator()(const LatticeArc &arc) {
    return StdArc(arc.ilabel, arc.olabel,
                  StdArc::Weight(arc.weight.Value1() + arc.weight.Value2()),
                  arc.nextstate);
  }
  MapFinalAction FinalAction() { return MAP_NO_SUPERFINAL; }

  MapSymbolsAction InputSymbolsAction() { return MAP_COPY_SYMBOLS; }

  MapSymbolsAction OutputSymbolsAction() { return MAP_COPY_SYMBOLS; }

  // I believe all properties are preserved.
  uint64 Properties(uint64 props) { return props; }
};


template<class Weight, class Int>
void PruneCompactLattice(
    Weight beam,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, Int> > > *fst);


} // end namespace fst

#include "fstext/lattice-utils-inl.h"

#endif  // KALDI_FSTEXT_LATTICE_WEIGHT_H_
