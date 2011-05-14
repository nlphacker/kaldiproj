// matrix/packed-matrix.h
// Copyright 2009-2011  Ondrej Glembek  Lukas Burget  Microsoft Corporation  Arnab Ghoshal  Yanmin Qian

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

#ifndef KALDI_MATRIX_PACKED_MATRIX_H_
#define KALDI_MATRIX_PACKED_MATRIX_H_

#include "matrix/matrix-common.h"
#include <algorithm>

namespace kaldi {

/// \addtogroup matrix_funcs_io
// we need to declare the friend << operator here
template<typename Real>
std::ostream & operator <<(std::ostream & out, const PackedMatrix<Real>& rM);


/// \addtogroup matrix_group
/// @{

/// @brief Packed matrix: base class for triangular and symmetric matrices.
template<typename Real> class PackedMatrix {
 public:
  PackedMatrix() : data_(NULL), num_rows_(0)
#ifdef KALDI_MEMALIGN_MANUAL
  , free_data_(NULL)
#endif
  {}

  explicit PackedMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero):
      data_(NULL) {  Resize(r, resize_type);  }

  explicit PackedMatrix(const PackedMatrix<Real>& rOrig) : data_(NULL) {
    Resize(rOrig.num_rows_);
    CopyFromPacked(rOrig);
  }

  template<class OtherReal>
  explicit PackedMatrix(const PackedMatrix<OtherReal>& rOrig) : data_(NULL) {
    Resize(rOrig.NumRows());
    CopyFromPacked(rOrig);
  }

  void SetZero();
  void SetUnit();  /// < Set to unit matrix.

  Real Trace() const;

  // Needed for inclusion in std::vector
  PackedMatrix<Real> & operator =(const PackedMatrix<Real> &other) {
    Resize(other.NumRows());
    CopyFromPacked(other);
    return *this;
  }

  ~PackedMatrix() {
    Destroy();
  }

  /// Set packed matrix to a specified size (can be zero).
  /// The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  /// This function takes time proportional to the number of data elements.
  void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero);

  void ScaleDiag(const Real alpha);  // Scales diagonal by alpha.

  template<class OtherReal>
  void CopyFromPacked(const PackedMatrix<OtherReal>& rOrig);

  Real* Data() { return data_; }
  const Real* Data() const { return data_; }
  inline MatrixIndexT NumRows() const { return num_rows_; }
  inline MatrixIndexT NumCols() const { return num_rows_; }
  MatrixIndexT SizeInBytes() const {
    return ((num_rows_ * (num_rows_ + 1)) / 2) * sizeof(Real);
  }

  // This code is duplicated in child classes to avoid extra levels of calls.
  Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                 static_cast<UnsignedMatrixIndexT>(c) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_)
                 && c <= r);
    return *(data_ + (r * (r + 1)) / 2 + c);
  }

  // This code is duplicated in child classes to avoid extra levels of calls.
  Real& operator() (MatrixIndexT r, MatrixIndexT c) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                 static_cast<UnsignedMatrixIndexT>(c) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_)
                 && c <= r);
    return *(data_ + (r * (r + 1)) / 2 + c);
  }

  Real Max() const {
    KALDI_ASSERT(num_rows_ > 0);
    return * (std::max_element(data_, data_ + ((num_rows_*(num_rows_+1))/2) ));
  }

  Real Min() const {
    KALDI_ASSERT(num_rows_ > 0);
    return * (std::min_element(data_, data_ + ((num_rows_*(num_rows_+1))/2) ));
  }


  // *this <-- *this + alpha* rV * rV^T.
  // The "2" in the name is because the argument is repeated.
  void AddVec2(const Real alpha, const Vector<Real>& rv);
  void Scale(Real c);

  friend std::ostream & operator << <> (std::ostream & out,
                                     const PackedMatrix<Real> &m);
  // Use instead of stream<<*this, if you want to add to existing contents.
  // Will throw exception on failure.
  void Read(std::istream & rIn, bool binary, bool add = false);

  void Write(std::ostream & rOut, bool binary) const;
  // binary = true is not yet supported.

  void Destroy();

  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(PackedMatrix<Real> *other);

 protected:
  // Will only be called from this class or derived classes.
  void AddPacked(const Real alpha, const PackedMatrix<Real>& rMa);
  Real* data_;
  MatrixIndexT num_rows_;
 private:
  /// Init assumes the current contents of the class are is invalid (i.e. junk or
  /// has already been freed), and it sets the matrixd to newly allocated memory
  /// with the specified dimension.  dim == 0 is acceptable.  The memory contents
  /// pointed to by data_ will be undefined.
  void Init(MatrixIndexT dim);
#ifdef KALDI_MEMALIGN_MANUAL
  Real* free_data_;
#endif


};
/// @} end "addtogroup matrix_group"


/// \addtogroup matrix_funcs_io
/// @{

template<typename Real>
std::ostream & operator << (std::ostream & os, const PackedMatrix<Real>& rM) {
  rM.Write(os, false);
  return os;
}

template<typename Real>
std::istream & operator >> (std::istream & is, PackedMatrix<Real> & rM) {
  rM.Read(is, false);
  return is;
}

/// @}

}  // namespace kaldi


// Including the implementation
#include "matrix/packed-matrix-inl.h"

#endif

