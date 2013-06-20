// tree/context-dep.h

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

#ifndef KALDI_TREE_CONTEXT_DEP_H_
#define KALDI_TREE_CONTEXT_DEP_H_

#include "itf/context-dep-itf.h"
#include "tree/event-map.h"
#include "matrix/matrix-lib.h"
#include "tree/cluster-utils.h"

/*
  This header provides the declarations for the class ContextDependency, which inherits
  from the interface class "ContextDependencyInterface" in itf/context-dep-itf.h.
  This is basically a wrapper around an EventMap.  The EventMap
  (tree/event-map.h) declares most of the internals of the class, and the building routines are
  in build-tree.h which uses build-tree-utils.h, which uses cluster-utils.h . */


namespace kaldi {

static const EventKeyType kPdfClass = -1;  // The "name" to which we assign the
// pdf-class (generally corresponds ot position in the HMM, zero-based);
// must not be used for any other event.  I.e. the value corresponding to
// this key is the pdf-class (see hmm-topology.h for explanation of what this is).


/* ContextDependency is quite a generic decision tree.

   It does not actually do very much-- all the magic is in the EventMap object.
   All this class does is to encode the phone context as a sequence of events, and
   pass this to the EventMap object to turn into what it will interpret as a
   vector of pdfs.

   Different versions of the ContextDependency class that are written in the future may
   have slightly different interfaces and pass more stuff in as events, to the
   EventMap object.

   In order to separate the process of training decision trees from the process
   of actually using them, we do not put any training code into the ContextDependency class.
 */
class ContextDependency: public ContextDependencyInterface {
 public:
  virtual int32 ContextWidth() const { return N_; }
  virtual int32 CentralPosition() const { return P_; }


  /// returns success or failure; outputs pdf to pdf_id
  virtual bool Compute(const std::vector<int32> &phoneseq,
                       int32 pdf_class, int32 *pdf_id) const;

  virtual int32 NumPdfs() const {
    // this routine could be simplified to return to_pdf_->MaxResult()+1.  we're a
    // bit more paranoid than that.
    if (!to_pdf_) return 0;
    EventAnswerType max_result = to_pdf_->MaxResult();
    if (max_result < 0 ) return 0;
    else return (int32) max_result+1;
  }
  virtual ContextDependencyInterface *Copy() const {
    return new ContextDependency(N_, P_, to_pdf_->Copy());
  }

  /// Read context-dependency object from disk; throws on error
  void Read (std::istream &is, bool binary);

  // Constructor with no arguments; will normally be called
  // prior to Read()
  ContextDependency(): N_(0), P_(0), to_pdf_(NULL) { }

  // Constructor takes ownership of pointers.
  ContextDependency(int32 N, int32 P,
                    EventMap *to_pdf):
      N_(N), P_(P), to_pdf_(to_pdf) { }
  void Write (std::ostream &os, bool binary) const;

  ~ContextDependency() { if (to_pdf_ != NULL) delete to_pdf_; }

  const EventMap &ToPdfMap() const { return *to_pdf_; }

  /// GetPdfInfo returns a vector indexed by pdf-id, saying for each pdf which
  /// pairs of (phone, pdf-class) it can correspond to.  (Usually just one).
  /// c.f. hmm/hmm-topology.h for meaning of pdf-class.

  void GetPdfInfo(const std::vector<int32> &phones,  // list of phones
                  const std::vector<int32> &num_pdf_classes,  // indexed by phone,
                  std::vector<std::vector<std::pair<int32, int32> > > *pdf_info)
      const;

 private:
  int32 N_;  //
  int32 P_;
  EventMap *to_pdf_;  // owned here.

  KALDI_DISALLOW_COPY_AND_ASSIGN(ContextDependency);
};

/// GenRandContextDependency is mainly of use for debugging.  Phones must be sorted and uniq
/// on input.
/// @param phones [in] A vector of phone id's [must be sorted and uniq].
/// @param ensure_all_covered [in] boolean argument; if true,  GenRandContextDependency
///        generates a context-dependency object that "works" for all phones [no gaps].
/// @param num_pdf_classes [out] outputs a vector indexed by phone, of the number
///          of pdf classes (e.g. states) for that phone.
/// @return Returns the a context dependency object.
ContextDependency *GenRandContextDependency(const std::vector<int32> &phones,
                                            bool ensure_all_covered,
                                            std::vector<int32> *num_pdf_classes);

/// GenRandContextDependencyLarge is like GenRandContextDependency but generates a larger tree
/// with specified N and P for use in "one-time" larger-scale tests.
ContextDependency *GenRandContextDependencyLarge(const std::vector<int32> &phones,
                                                 int N, int P,
                                                 bool ensure_all_covered,
                                                 std::vector<int32> *num_pdf_classes);

// MonophoneContextDependency() returns a new ContextDependency object that
// corresponds to a monophone system.
// The map phone2num_pdf_classes maps from the phone id to the number of
// pdf-classes we have for that phone (e.g. 3, so the pdf-classes would be
// 0, 1, 2).

ContextDependency*
MonophoneContextDependency(const std::vector<int32> phones,
                           const std::vector<int32> phone2num_pdf_classes);

// MonophoneContextDependencyShared is as MonophoneContextDependency but lets
// you define classes of phones which share pdfs (e.g. different stress-markers of a single
// phone.)  Each element of phone_classes is a set of phones that are in that class.
ContextDependency*
MonophoneContextDependencyShared(const std::vector<std::vector<int32> > phone_classes,
                                 const std::vector<int32> phone2num_pdf_classes);


// Important note:
// Statistics for training decision trees will be of type:
// std::vector<std::pair<EventType, Clusterable*> >
// We don't make this a typedef as it doesn't add clarity.
// they will be sorted and unique on the EventType member, which
// itself is sorted and unique on the name (see event-map.h).

// See build-tree.h for functions relating to actually building the decision trees.




}  // namespace Kaldi


#endif
