// matrix/tp-matrix.h

// Copyright 2009-2011  Ondrej Glembek;  Lukas Burget;  Microsoft Corporation;
//                      Saarland University;  Yanmin Qian


// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_MATRIX_TP_MATRIX_H_
#define KALDI_MATRIX_TP_MATRIX_H_


#include "matrix/packed-matrix.h"

namespace kaldi {
/// \addtogroup matrix_group
/// @{

template<typename Real> class TpMatrix;

/// @brief Packed symetric matrix class
template<typename Real>
class TpMatrix : public PackedMatrix<Real> {
 public:
  TpMatrix() : PackedMatrix<Real>() {}
  explicit TpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
      : PackedMatrix<Real>(r, resize_type) {}
  TpMatrix(const TpMatrix<Real>& Orig) : PackedMatrix<Real>(Orig) {}
  template<class OtherReal> explicit TpMatrix(const TpMatrix<OtherReal>& Orig)
      : PackedMatrix<Real>(Orig) {}
  ~TpMatrix() {}

  Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    if (static_cast<UnsignedMatrixIndexT>(c) >
        static_cast<UnsignedMatrixIndexT>(r)) {
      KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(c) <
                   static_cast<UnsignedMatrixIndexT>(this->num_rows_));
      return 0;
    }
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(this->num_rows_));
    // c<=r now so don't have to check c.
    return *(this->data_ + (r*(r+1)) / 2 + c);
    // Duplicating code from PackedMatrix.h
  }

  Real& operator() (MatrixIndexT r, MatrixIndexT c) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(this->num_rows_));
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(c) <=
                 static_cast<UnsignedMatrixIndexT>(r) &&
                 "you cannot access the upper triangle of TpMatrix using "
                 "a non-const matrix object.");
    return *(this->data_ + (r*(r+1)) / 2 + c);
    // Duplicating code from PackedMatrix.h
  }

  void Cholesky(const SpMatrix<Real>& Orig);
  void Invert();
  // Inverts in double precision.
  void InvertDouble() {
    TpMatrix<double> dmat(*this);
    dmat.Invert();
    (*this).CopyFromTp(dmat);
  }

  /// Shallow swap
  void Swap(TpMatrix<Real> *other);

  /// Returns the determinant of the matrix (product of diagonals)
  Real Determinant();

  /// CopyFromMat copies the lower triangle of M into *this
  /// (or the upper triangle, if Trans == kTrans).
  void CopyFromMat(MatrixBase<Real> &M,
                   MatrixTransposeType Trans = kNoTrans);

  /// CopyFromTp copies andother triangular matrix into this one.
  void CopyFromTp(const TpMatrix<Real> &other) {
    PackedMatrix<Real>::CopyFromPacked(other);
  }

  template<class OtherReal> void CopyFromTp(const TpMatrix<OtherReal> &other) {
    PackedMatrix<Real>::CopyFromPacked(other);
  }

  /// AddTp does *this += alpha * M.
  void AddTp(const Real alpha, const TpMatrix<Real> &M) {
    this->AddPacked(alpha, M);
  }

  using PackedMatrix<Real>::operator =;
  using PackedMatrix<Real>::Scale;

  void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero) {
    PackedMatrix<Real>::Resize(nRows, resize_type);
  }
};

/// @} end of "addtogroup matrix_group".

}  // namespace kaldi


// Including the implementation
#include "matrix/tp-matrix-inl.h"


#endif
