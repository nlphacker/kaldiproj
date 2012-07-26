// transform/basis-fmllr-diag-gmm.h

// Copyright 2012  Carnegie Mellon University (author: Yajie Miao)

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


#ifndef KALDI_BASIS_FMLLR_DIAG_GMM_H_
#define KALDI_BASIS_FMLLR_DIAG_GMM_H_

#include <vector>
#include <string>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/mle-full-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "transform/transform-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"

namespace kaldi {

/* This header contains routines for performing subspace CMLLR
   (without a regression tree) for diagonal GMM acoustic model.

   Refer to Dan Povey's paper for derivations:
   Daniel Povey, Kaisheng Yao. A basis representation of constrained
   MLLR transforms for robust adaptation. Computer Speech and Language,
   volume 26:35–51, 2012.
*/

struct BasisFmllrOptions {
  int32 num_iters;
  BaseFloat size_scale;
  BaseFloat min_count;
  int32 step_size_iters;
  BasisFmllrOptions(): num_iters(10), size_scale(0.2), min_count(500.0), step_size_iters(3) { }
  void Register(ParseOptions *po) {
    po->Register("num-iters", &num_iters,
                 "Number of iterations in basis fMLLR update during testing");
    po->Register("size-scale", &size_scale,
                 "Scale (< 1.0) of speaker occupancy to decide base number");
    po->Register("fmllr-min-count", &min_count,
                 "Minimum count required to update fMLLR");
    po->Register("step-size-iters", &step_size_iters,
                 "Number of iterations in computing step size");
  }
};

/** \class BasisFmllrAccus
 *  Stats for fMLLR subspace estimation.  This class is only to estimate
 *  the "basis", which is done in training time.  Class BasisFmllrEstimate
 *  contains the functions that are used in test time.  (see the
 *  function BasisFmllrCoefficients()).
 */
class BasisFmllrAccus {

 public:
  BasisFmllrAccus() { }
  explicit BasisFmllrAccus(int32 dim) {
	  dim_ = dim;
	  ResizeAccus(dim);
  }

  void ResizeAccus(int32 dim);

  /// Routines for reading and writing stats
  void Write(std::ostream &out_stream, bool binary) const;
  void Read(std::istream &in_stream, bool binary, bool add = false);

  /// Accumulate gradient scatter for one (training) speaker.
  /// To finish the process, we need to traverse the whole training
  /// set. Parallelization works if the speaker list is splitted, and
  /// stats are summed up by setting add=true in BasisFmllrEstimate::
  /// ReadBasis. See section 5.2 of the paper.
  void AccuGradientScatter(const AffineXformStats &spk_stats);

  /// Gradient scatter. Dim is [(D+1)*D] [(D+1)*D]
  SpMatrix<BaseFloat> grad_scatter_;
  /// Feature dimension
  int32 dim_;
};

/** \class BasisFmllrEstimate
 *  Estimation functions for basis fMLLR.
 */
class BasisFmllrEstimate {

 public:
  BasisFmllrEstimate() { }
  explicit BasisFmllrEstimate(int32 dim) {
	  dim_ = dim; basis_size_ = dim * (dim + 1);
  }

  /// Routines for reading and writing fMLLR base matrices
  void WriteBasis(std::ostream &out_stream, bool binary) const;
  void ReadBasis(std::istream &in_stream, bool binary, bool add = false);

  /// Estimate the base matrices efficiently in a Maximum Likelihood manner.
  /// It takes diagonal GMM as argument, which will be used for preconditioner
  /// computation. The total number of bases is fixed to
  /// N = (dim + 1) * dim
  /// Note that SVD is performed in the normalized space. The base matrices
  /// are finally converted back to the unnormalized space.
  void EstimateFmllrBasis(const AmDiagGmm &am_gmm,
                          const BasisFmllrAccus &basis_accus);

  /// This function computes the preconditioner matrix, prior to base matrices
  /// estimation. Since the expected values of G statistics are used, it
  /// takes the acoustic model as the argument, rather than the actual
  /// accumulations AffineXformStats
  /// See section 5.1 of the paper.
  void ComputeAmDiagPrecond(const AmDiagGmm &am_gmm,
                            SpMatrix<double> *pre_cond);

  /// This function performs speaker adaptation, computing the fMLLR matrix
  /// based on speaker statistics. It takes fMLLR stats as argument.
  /// The basis weights (d_{1}, d_{2}, ..., d_{N}) are also optimized
  /// explicitly. Finally, it returns objective function improvement over
  /// all the iterations.
  /// See section 5.3 of the paper for more details.
  double BasisFmllrCoefficients(const AffineXformStats &spk_stats,
  	                            Matrix<BaseFloat> *out_xform,
  	                            Vector<BaseFloat> *coefficient,
    	                        BasisFmllrOptions options);

  /// Basis matrices. Dim is [T] [D] [D+1]
  /// T is the number of bases
  vector< Matrix<BaseFloat> > fmllr_basis_;
  /// Feature dimension
  int32 dim_;
  /// Number of bases D*(D+1)
  int32 basis_size_;

};


/// This function takes the step direction (delta) of fMLLR matrix as argument,
/// and optimize step size using Newton's method. This is an iterative method,
/// where each iteration should not decrease the auxiliary function. Note that
/// the resulting step size \k should be close to 1. If \k <<1 or >>1, there
/// maybe problems with preconditioning or the speaker stats.
double CalBasisFmllrStepSize(const AffineXformStats &spk_stats,
                             const Matrix<double> &delta,
                             const Matrix<double> &A,
                             const Matrix<double> &S,
                             int32 max_iters);

} // namespace kaldi

#endif  // KALDI_BASIS_FMLLR_DIAG_GMM_H_
