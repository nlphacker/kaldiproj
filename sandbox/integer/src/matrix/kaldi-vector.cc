// matrix/kaldi-vector.cc

// Copyright 2009-2011   Microsoft Corporation;  Lukas Burget;
//                      Saarland University;   Go Vivace Inc.;  Ariya Rastrow;
//                      Petr Schwarz;  Yanmin Qian;  Jan Silovsky


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
#include <string>

#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/sp-matrix.h"

namespace kaldi {

template<> float VecVec<>(const VectorBase<float> &ra,
                          const VectorBase<float> &rb) {
  MatrixIndexT adim = ra.Dim();
  KALDI_ASSERT(adim == rb.Dim());
  return cblas_sdot(adim, ra.Data(), 1, rb.Data(), 1);
}

template<> double VecVec<>(const VectorBase<double> &ra,
                           const VectorBase<double> &rb) {
  MatrixIndexT adim = ra.Dim();
  KALDI_ASSERT(adim == rb.Dim());
  return cblas_ddot(adim, ra.Data(), 1, rb.Data(), 1);
}


template<class Real, class OtherReal>
Real VecVec(const VectorBase<Real> &ra,
            const VectorBase<OtherReal> &rb) {
  MatrixIndexT adim = ra.Dim();
  KALDI_ASSERT(adim == rb.Dim());
  const Real *a_data = ra.Data();
  const OtherReal *b_data = rb.Data();
  Real sum = 0.0;
  for (MatrixIndexT i = 0; i < adim; i++)
    sum += a_data[i]*b_data[i];
  return sum;
}

// instantiate the template above.
template
float VecVec<>(const VectorBase<float> &ra,
               const VectorBase<double> &rb);
template
double VecVec<>(const VectorBase<double> &ra,
                const VectorBase<float> &rb);


template<>
template<>
void VectorBase<float>::AddVec(const float alpha,
                               const VectorBase<float> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
  KALDI_ASSERT(&v != this);
  cblas_saxpy(dim_, alpha, v.Data(), 1, data_, 1);
}

template<>
template<>
void VectorBase<double>::AddVec(const double alpha,
                                const VectorBase<double> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
  KALDI_ASSERT(&v != this);
  cblas_daxpy(dim_, alpha, v.Data(), 1, data_, 1);
}


template<>
void VectorBase<float>::AddMatVec(const float alpha,
                                  const MatrixBase<float> &M,
                                  MatrixTransposeType trans,
                                  const VectorBase<float> &v,
                                  const float beta) {
  KALDI_ASSERT((trans == kNoTrans && M.NumCols() == v.dim_ && M.NumRows() == dim_)
               || (trans == kTrans && M.NumRows() == v.dim_ && M.NumCols() == dim_));
  KALDI_ASSERT(&v != this);
  cblas_sgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), M.NumRows(),
              M.NumCols(), alpha, M.Data(), M.Stride(), v.Data(), 1, beta,
              data_, 1);
}

template<>
void VectorBase<double>::AddMatVec(const double alpha,
                                   const MatrixBase<double> &M,
                                   MatrixTransposeType trans,
                                   const VectorBase<double> &v,
                                   const double beta) {
  KALDI_ASSERT((trans == kNoTrans && M.NumCols() == v.dim_ && M.NumRows() == dim_)
               || (trans == kTrans && M.NumRows() == v.dim_ && M.NumCols() == dim_));
  KALDI_ASSERT(&v != this);
  cblas_dgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), M.NumRows(),
              M.NumCols(), alpha, M.Data(), M.Stride(), v.Data(), 1, beta,
              data_, 1);
}


template<>
void VectorBase<float>::AddSpVec(const float alpha,
                                 const SpMatrix<float> &M,
                                 const VectorBase<float> &v,
                                 const float beta) {
  KALDI_ASSERT(M.NumRows() == v.dim_ && dim_ == v.dim_);
  KALDI_ASSERT(&v != this);
  cblas_sspmv(CblasRowMajor, CblasLower, M.NumRows(), alpha, M.Data(),
              v.Data(), 1, beta, data_, 1);
}

template<>
void VectorBase<double>::AddSpVec(const double alpha,
                                  const SpMatrix<double> &M,
                                  const VectorBase<double> &v,
                                  const double beta) {
  KALDI_ASSERT(M.NumRows() == v.dim_ && dim_ == v.dim_);
  KALDI_ASSERT(&v != this);
  cblas_dspmv(CblasRowMajor, CblasLower, M.NumRows(), alpha, M.Data(),
              v.Data(), 1, beta, data_, 1);
}

template<>
void VectorBase<float>::MulTp(const TpMatrix<float> &M,
                              const MatrixTransposeType trans) {
  KALDI_ASSERT(M.NumRows() == dim_);
  cblas_stpmv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, M.NumRows(), M.Data(), data_, 1);
}

template<>
void VectorBase<double>::MulTp(const TpMatrix<double> &M,
                               const MatrixTransposeType trans) {
  KALDI_ASSERT(M.NumRows() == dim_);
  cblas_dtpmv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, M.NumRows(),  M.Data(), data_, 1);
}

template<typename Real>
inline void Vector<Real>::Init(const MatrixIndexT dim) {
  if (dim == 0) {
    this->dim_ = 0;
    this->data_ = NULL;
#ifdef KALDI_MEMALIGN_MANUAL
    free_data_ = NULL;
#endif
    return;
  }
  MatrixIndexT size;
  void *data;
  void *free_data;

  // size = align<16>(dim * sizeof(Real));
  size = dim * sizeof(Real);

  if ((data = KALDI_MEMALIGN(16, size, &free_data)) != NULL) {
    this->data_ = static_cast<Real*> (data);
#ifdef KALDI_MEMALIGN_MANUAL
    free_data_ = static_cast<Real*> (free_data);
#endif
    this->dim_ = dim;
  } else {
    throw std::bad_alloc();
  }
}


template<typename Real>
void Vector<Real>::Resize(const MatrixIndexT dim, MatrixResizeType resize_type) {

  // the next block uses recursion to handle what we have to do if
  // resize_type == kCopyData.
  if (resize_type == kCopyData) {
    if (this->data_ == NULL || dim == 0) resize_type = kSetZero;  // nothing to copy.
    else if (this->dim_ == dim) { return; } // nothing to do.
    else {
      // set tmp to a vector of the desired size.
      Vector<Real> tmp(dim, kUndefined);
      if (dim > this->dim_) {
        memcpy(tmp.data_, this->data_, sizeof(Real)*this->dim_);
        memset(tmp.data_+this->dim_, 0, sizeof(Real)*(dim-this->dim_));
      } else {
        memcpy(tmp.data_, this->data_, sizeof(Real)*dim);
      }
      tmp.Swap(this);
      // and now let tmp go out of scope, deleting what was in *this.
      return;
    }
  }
  // At this point, resize_type == kSetZero or kUndefined.

  if (this->data_ != NULL) {
    if (this->dim_ == dim) {
      if (resize_type == kSetZero) this->SetZero();
      return;
    } else {
      Destroy();
    }
  }
  Init(dim);
  if (resize_type == kSetZero) this->SetZero();
}


/// Copy data from another vector
template<typename Real>
void VectorBase<Real>::CopyFromVec(const VectorBase<Real> &v) {
  KALDI_ASSERT(Dim() == v.Dim());
  CopyFromPtr(v.data_, v.Dim());
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::CopyFromPacked(const PackedMatrix<OtherReal>& M) {
  SubVector<OtherReal> v(M);
  this->CopyFromVec(v);
}
// instantiate the template.
template void VectorBase<float>::CopyFromPacked(const PackedMatrix<double> &other);
template void VectorBase<float>::CopyFromPacked(const PackedMatrix<float> &other);
template void VectorBase<double>::CopyFromPacked(const PackedMatrix<double> &other);
template void VectorBase<double>::CopyFromPacked(const PackedMatrix<float> &other);


/// Load data into the vector
template<typename Real>
void VectorBase<Real>::CopyFromPtr(const Real* Data, MatrixIndexT sz) {
  KALDI_ASSERT(dim_ == sz);
  std::memcpy(this->data_, Data, Dim() * sizeof(Real));
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::CopyFromVec(const VectorBase<OtherReal> &other) {
  KALDI_ASSERT(dim_ == other.Dim());
  const OtherReal *other_ptr = other.Data();
  for (MatrixIndexT i = 0; i < dim_; i++) { data_[i] = other_ptr[i]; }
}

template void VectorBase<float>::CopyFromVec(const VectorBase<double> &other);
template void VectorBase<double>::CopyFromVec(const VectorBase<float> &other);

// Remove element from the vector. The vector is non reallocated
template<typename Real>
void Vector<Real>::RemoveElement(MatrixIndexT i) {
  KALDI_ASSERT(i <  this->dim_ && "Access out of vector");
  for (MatrixIndexT j = i + 1; j <  this->dim_; j++)
    this->data_[j-1] =  this->data_[j];
  this->dim_--;
}


/// Deallocates memory and sets object to empty vector.
template<typename Real>
void Vector<Real>::Destroy() {
  /// we need to free the data block if it was defined
#ifndef KALDI_MEMALIGN_MANUAL
  if (this->data_ != NULL)
    KALDI_MEMALIGN_FREE(this->data_);
#else
  if (this->data_ != NULL)
    KALDI_MEMALIGN_FREE(free_data_);
  free_data_ = NULL;
#endif
  this->data_ = NULL;
  this->dim_ = 0;
}

template<typename Real>
void VectorBase<Real>::SetZero() {
  std::memset(data_, 0, dim_ * sizeof(Real));
}

template<typename Real>
bool VectorBase<Real>::IsZero(Real cutoff) const {
  Real abs_max = 0.0;
  for (MatrixIndexT i = 0; i < Dim(); i++)
    abs_max = std::max(std::abs(data_[i]), abs_max);
  return (abs_max <= cutoff);
}

template<typename Real>
void VectorBase<Real>::SetRandn() {
  for (MatrixIndexT i = 0; i < Dim(); i++) data_[i] = kaldi::RandGauss();
}


template<typename Real>
void VectorBase<Real>::Set(Real f) {
  // Why not use memset here?
  for (MatrixIndexT i = 0; i < dim_; i++) { data_[i] = f; }
}

template<typename Real>
void VectorBase<Real>::CopyRowsFromMat(const MatrixBase<Real> &mat) {
  KALDI_ASSERT(dim_ == mat.NumCols() * mat.NumRows());

  Real *inc_data = data_;
  const MatrixIndexT cols = mat.NumCols(), rows = mat.NumRows();

  if (mat.Stride() == mat.NumCols()) {
    memcpy(inc_data, mat.Data(), cols*rows*sizeof(Real));
  } else {
    for (MatrixIndexT i = 0; i < rows; i++) {
      // copy the data to the propper position
      memcpy(inc_data, mat.RowData(i), cols * sizeof(Real));
      // set new copy position
      inc_data += cols;
    }
  }
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::CopyRowsFromMat(const MatrixBase<OtherReal> &mat) {
  KALDI_ASSERT(dim_ == mat.NumCols() * mat.NumRows());
  Real *vec_data = data_;
  const MatrixIndexT cols = mat.NumCols(),
      rows = mat.NumRows();

  for (MatrixIndexT i = 0; i < rows; i++) {
    const OtherReal *mat_row = mat.RowData(i);
    for (MatrixIndexT j = 0; j < cols; j++) {
      vec_data[j] = static_cast<Real>(mat_row[j]);
    }
    vec_data += cols;
  }
}

template
void VectorBase<float>::CopyRowsFromMat(const MatrixBase<double> &mat);
template
void VectorBase<double>::CopyRowsFromMat(const MatrixBase<float> &mat);


template<typename Real>
void VectorBase<Real>::CopyColsFromMat(const MatrixBase<Real> &mat) {
  KALDI_ASSERT(dim_ == mat.NumCols() * mat.NumRows());

  Real*       inc_data = data_;
  const MatrixIndexT  cols     = mat.NumCols(), rows = mat.NumRows(), stride = mat.Stride();
  const Real *mat_inc_data = mat.Data();

  for (MatrixIndexT i = 0; i < cols; i++) {
    for (MatrixIndexT j = 0; j < rows; j++) {
      inc_data[j] = mat_inc_data[j*stride];
    }
    mat_inc_data++;
    inc_data += rows;
  }
}

template<typename Real>
void VectorBase<Real>::CopyRowFromMat(const MatrixBase<Real> &mat, MatrixIndexT row) {
  KALDI_ASSERT(row < mat.NumRows());
  KALDI_ASSERT(dim_ == mat.NumCols());
  const Real *mat_row = mat.RowData(row);
  memcpy(data_, mat_row, sizeof(Real)*dim_);
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::CopyRowFromMat(const MatrixBase<OtherReal> &mat, MatrixIndexT row) {
  KALDI_ASSERT(row < mat.NumRows());
  KALDI_ASSERT(dim_ == mat.NumCols());
  const OtherReal *mat_row = mat.RowData(row);
  for (MatrixIndexT i = 0; i < dim_; i++)
    data_[i] = static_cast<Real>(mat_row[i]);
}

template
void VectorBase<float>::CopyRowFromMat(const MatrixBase<double> &mat, MatrixIndexT row);
template
void VectorBase<double>::CopyRowFromMat(const MatrixBase<float> &mat, MatrixIndexT row);

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::CopyRowFromSp(const SpMatrix<OtherReal> &sp, MatrixIndexT row) {
  KALDI_ASSERT(row < sp.NumRows());
  KALDI_ASSERT(dim_ == sp.NumCols());
  
  const OtherReal *sp_data = sp.Data();

  sp_data += (row*(row+1)) / 2; // takes us to beginning of this row.
  MatrixIndexT i;
  for (i = 0; i < row; i++) // copy consecutive elements.
    data_[i] = static_cast<Real>(*(sp_data++));
  for(; i < dim_; ++i, sp_data += i) 
    data_[i] = static_cast<Real>(*sp_data);
}

template
void VectorBase<float>::CopyRowFromSp(const SpMatrix<double> &mat, MatrixIndexT row);
template
void VectorBase<double>::CopyRowFromSp(const SpMatrix<float> &mat, MatrixIndexT row);
template
void VectorBase<float>::CopyRowFromSp(const SpMatrix<float> &mat, MatrixIndexT row);
template
void VectorBase<double>::CopyRowFromSp(const SpMatrix<double> &mat, MatrixIndexT row);


// takes elements to a power.  Throws exception if could not (but only for power != 1 ad power != 2).
template<typename Real>
void VectorBase<Real>::ApplyPow(Real power) {
  if (power == 1.0) return;
  if (power == 2.0) {
    for (MatrixIndexT i = 0; i < dim_; i++)
      data_[i] = data_[i] * data_[i];
  } else if (power == 0.5) {
    for (MatrixIndexT i = 0; i < dim_; i++) {
      if (!(data_[i] >= 0.0))
        KALDI_ERR << "Cannot take square root of negative value "
                  << data_[i];
      data_[i] = sqrt(data_[i]);
    }
  } else {
    for (MatrixIndexT i = 0; i < dim_; i++) {
      data_[i] = pow(data_[i], power);
      if (data_[i] == HUGE_VAL) {  // HUGE_VAL is what errno returns on error.
        KALDI_ERR << "Could not raise element "  << i << "to power "
                  << power << ": returned value = " << data_[i];
      }
    }
  }
}

// Computes the p-th norm. Throws exception if could not.
template<typename Real>
Real VectorBase<Real>::Norm(Real p) const {
  KALDI_ASSERT(p >= 0.0);
  Real sum = 0.0;
  if (p == 0.0) {
    for (MatrixIndexT i = 0; i < dim_; i++)
      if (data_[i] != 0.0) sum += 1.0;
    return sum;
  } else if (p == 1.0) {
    for (MatrixIndexT i = 0; i < dim_; i++)
      sum += std::abs(data_[i]);
    return sum;
  } else if (p == 2.0) {
    for (MatrixIndexT i = 0; i < dim_; i++)
      sum += data_[i] * data_[i];
    return sqrt(sum);
  } else {
    Real tmp;
    for (MatrixIndexT i = 0; i < dim_; i++) {
      tmp = pow(std::abs(data_[i]), p);
      if (tmp == HUGE_VAL) {  // HUGE_VAL is what pow returns on error.
        KALDI_ERR << "Could not raise element " << i << "to power " << p
                  << ": returned value = " << tmp;
      }
      sum += tmp;
    }
    tmp = pow(sum, static_cast<Real>(1.0/p));
    if (tmp == HUGE_VAL) {  // HUGE_VAL is what errno returns on error.
      KALDI_ERR << "Could not take the " << p << "-th root of " << sum
                << "; returned value = " << tmp;
    }
    return tmp;
  }
}

template<typename Real>
bool VectorBase<Real>::ApproxEqual(const VectorBase<Real> &other, float tol) const {
  if (dim_ != other.dim_) KALDI_ERR << "ApproxEqual: size mismatch "
                                    << dim_ << " vs. " << other.dim_;
  KALDI_ASSERT(tol >= 0.0);
  if (tol != 0.0) {
    Vector<Real> tmp(*this);
    tmp.AddVec(-1.0, other);
    return (tmp.Norm(2.0) <= static_cast<Real>(tol) * this->Norm(2.0));
  } else { // Test for exact equality.
    const Real *data = data_;
    const Real *other_data = other.data_;
    for (MatrixIndexT dim = dim_, i = 0; i < dim; i++)
      if (data[i] != other_data[i]) return false;
    return true;
  }
}

template<typename Real>
Real VectorBase<Real>::Max() const {
  if (dim_ == 0) KALDI_ERR << "Empty vector";
  Real ans = data_[0];
  for (MatrixIndexT i = 1; i < dim_; i++) { ans = std::max(ans, data_[i]); }
  return ans;
}

template<typename Real>
Real VectorBase<Real>::Min() const {
  if (dim_ == 0) KALDI_ERR << "Empty vector";
  Real ans = data_[0];
  for (MatrixIndexT i = 1; i < dim_; i++) { ans = std::min(ans, data_[i]); }
  return ans;
}


template<typename Real>
void VectorBase<Real>::CopyColFromMat(const MatrixBase<Real> &mat, MatrixIndexT col) {
  KALDI_ASSERT(col < mat.NumCols());
  KALDI_ASSERT(dim_ == mat.NumRows());
  for (MatrixIndexT i = 0; i < dim_; i++)
    data_[i] = mat(i, col);
  // can't do this very efficiently so don't really bother. could improve this though.
}

template<typename Real>
void VectorBase<Real>::CopyDiagFromMat(const MatrixBase<Real> &M) {
  KALDI_ASSERT(dim_ == std::min(M.NumRows(), M.NumCols()));
  for (MatrixIndexT i = 0; i < dim_; i++)
    data_[i] = M(i, i);
  // could make this more efficient.
}

template<typename Real>
void VectorBase<Real>::CopyDiagFromPacked(const PackedMatrix<Real> &M) {
  KALDI_ASSERT(dim_ == M.NumCols());
  for (MatrixIndexT i = 0; i < dim_; i++)
    data_[i] = M(i, i);
  // could make this more efficient.
}

template<typename Real>
Real VectorBase<Real>::Sum() const {
  double sum = 0.0;
  for (MatrixIndexT i = 0; i < dim_; i++) { sum += data_[i]; }
  return sum;
}

template<typename Real>
Real VectorBase<Real>::SumLog() const {
  double sum_log = 0.0;
  double prod = 1.0;
  for (MatrixIndexT i = 0; i < dim_; i++) {
    prod *= data_[i];
    // Possible future work (arnab): change these magic values to pre-defined
    // constants
    if (prod < 1.0e-10 || prod > 1.0e+10) {  
      sum_log += log(prod);
      prod = 1.0;
    }
  }
  if (prod != 1.0) sum_log += log(prod);
  return sum_log;
}

template<typename Real>
void VectorBase<Real>::AddRowSumMat(Real alpha, const MatrixBase<Real> &M, Real beta) {
  // note the double accumulator
  KALDI_ASSERT(dim_ == M.NumCols());
  MatrixIndexT num_rows = M.NumRows(), stride = M.Stride();
  for (MatrixIndexT i = 0; i < dim_; i++) {
    double sum = 0.0;
    const Real *src = M.Data() + i;
    for (MatrixIndexT j = 0; j < num_rows; j++)
      sum += src[j*stride];
    data_[i] = alpha * sum + beta * data_[i];
  }
}

template<typename Real>
void VectorBase<Real>::AddColSumMat(Real alpha, const MatrixBase<Real> &M, Real beta) {
  // note the double accumulator
  double sum;
  KALDI_ASSERT(dim_ == M.NumRows());
  MatrixIndexT num_cols = M.NumCols();
  for (MatrixIndexT i = 0; i < dim_; i++) {
    sum = 0.0;
    const Real *src = M.RowData(i);
    for (MatrixIndexT j = 0; j < num_cols; j++)
      sum += src[j];
    data_[i] = alpha * sum + beta * data_[i];;
  }
}

template<typename Real>
Real VectorBase<Real>::LogSumExp(Real prune) const {
  Real sum;
  if (sizeof(sum) == 8) sum = kLogZeroDouble;
  else sum = kLogZeroFloat;
  Real max_elem = Max(), cutoff;
  if (sizeof(Real) == 4) cutoff = max_elem + kMinLogDiffFloat;
  else cutoff = max_elem + kMinLogDiffDouble;
  if (prune > 0.0 && max_elem - prune > cutoff) // explicit pruning...
    cutoff = max_elem - prune;

  double sum_relto_max_elem = 0.0;

  for (MatrixIndexT i = 0; i < dim_; i++) {
    BaseFloat f = data_[i];
    if (f >= cutoff)
      sum_relto_max_elem += exp(f - max_elem);
  }
  return max_elem + std::log(sum_relto_max_elem);
}

template<typename Real>
void VectorBase<Real>::InvertElements() {
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] = static_cast<Real>(1 / data_[i]);
  }
}

template<typename Real>
void VectorBase<Real>::ApplyLog() {
  for (MatrixIndexT i = 0; i < dim_; i++) {
    if (data_[i] < 0.0)
      KALDI_ERR << "Trying to take log of a negative number.";
    data_[i] = log(data_[i]);
  }
}

template<typename Real>
void VectorBase<Real>::ApplyLogAndCopy(const VectorBase<Real> &v) {
  KALDI_ASSERT(dim_ == v.Dim());
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] = log(v(i));
  }
}

template<typename Real>
void VectorBase<Real>::ApplyExp() {
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] = exp(data_[i]);
  }
}

template<typename Real>
void VectorBase<Real>::Abs() {
  for (MatrixIndexT i = 0; i < dim_; i++) { data_[i] = std::abs(data_[i]); }
}

template<typename Real>
MatrixIndexT VectorBase<Real>::ApplyFloor(Real floor_val) {
  MatrixIndexT num_floored = 0;
  for (MatrixIndexT i = 0; i < dim_; i++) {
    if (data_[i] < floor_val) {
      data_[i] = floor_val;
      num_floored++;
    }
  }
  return num_floored;
}

template<typename Real>
MatrixIndexT VectorBase<Real>::ApplyFloor(const VectorBase<Real> &floor_vec) {
  KALDI_ASSERT(floor_vec.Dim() == dim_);
  MatrixIndexT num_floored = 0;
  for (MatrixIndexT i = 0; i < dim_; i++) {
    if (data_[i] < floor_vec(i)) {
      data_[i] = floor_vec(i);
      num_floored++;
    }
  }
  return num_floored;
}

template<typename Real>
Real VectorBase<Real>::ApplySoftMax() {
  Real max = this->Max(), sum = 0.0;
  for (MatrixIndexT i = 0; i < dim_; i++) {
    sum += (data_[i] = exp(data_[i] - max));
  }
  this->Scale(1.0 / sum);
  return max + log(sum);
}


template<typename Real>
void VectorBase<Real>::Add(Real c) {
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] += c;
  }
}

template<>
void VectorBase<float>::Scale(float alpha) {
  cblas_sscal(dim_, alpha, data_, 1);
}

template<>
void VectorBase<double>::Scale(double alpha) {
  cblas_dscal(dim_, alpha, data_, 1);
}

template<typename Real>
void VectorBase<Real>::MulElements(const VectorBase<Real> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] *= v.data_[i];
  }
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::MulElements(const VectorBase<OtherReal> &v) {
  KALDI_ASSERT(dim_ == v.Dim());
  const OtherReal *other_ptr = v.Data();
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] *= other_ptr[i];
  }
}
// instantiate template.
template
void VectorBase<float>::MulElements(const VectorBase<double> &v);
template
void VectorBase<double>::MulElements(const VectorBase<float> &v);

template<typename Real>
void VectorBase<Real>::AddVecVec(Real alpha, const VectorBase<Real> &v,
                                 const VectorBase<Real> &r, Real beta) {
  KALDI_ASSERT((dim_ == v.dim_ && dim_ == r.dim_));
  KALDI_ASSERT(this != &v && this != &r);
  // remove __restrict__ if it causes compilation problems.
  register __restrict__  Real *data = data_, *vdata = v.data_, *rdata = r.data_;

  if (beta == 1.0) {
    for (MatrixIndexT i = 0; i < dim_; i++)
      data[i] += alpha * vdata[i] * rdata[i];
  } else if (beta== 0.0) {
    for (MatrixIndexT i = 0; i < dim_; i++)
      data[i] = alpha * vdata[i] * rdata[i];
  } else {
    for (MatrixIndexT i = 0; i < dim_; i++)
      data[i] = alpha * vdata[i] * rdata[i] + beta * data[i];
  }
}

template<typename Real>
void VectorBase<Real>::DivElements(const VectorBase<Real> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] /= v.data_[i];
  }
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::DivElements(const VectorBase<OtherReal> &v) {
  KALDI_ASSERT(dim_ == v.Dim());
  const OtherReal *other_ptr = v.Data();
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] /= other_ptr[i];
  }
}
// instantiate template.
template
void VectorBase<float>::DivElements(const VectorBase<double> &v);
template
void VectorBase<double>::DivElements(const VectorBase<float> &v);

template<typename Real>
void VectorBase<Real>::AddVecDivVec(Real alpha, const VectorBase<Real> &v,
                                    const VectorBase<Real> &rr, Real beta) {
  KALDI_ASSERT((dim_ == v.dim_ && dim_ == rr.dim_));
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] = alpha * v.data_[i]/rr.data_[i] + beta * data_[i] ;
  }
}

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::AddVec(const Real alpha, const VectorBase<OtherReal> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
  // remove __restrict__ if it causes compilation problems.
  register __restrict__  Real *data = data_;
  register __restrict__  OtherReal *other_data = v.data_;
  MatrixIndexT dim = dim_;
  if (alpha != 1.0)
    for (MatrixIndexT i = 0; i < dim; i++)
      data[i] += alpha*other_data[i];
  else
    for (MatrixIndexT i = 0; i < dim; i++)
      data[i] += other_data[i];
}

template
void VectorBase<float>::AddVec(const float alpha, const VectorBase<double> &v);
template
void VectorBase<double>::AddVec(const double alpha, const VectorBase<float> &v);

template<typename Real>
template<typename OtherReal>
void VectorBase<Real>::AddVec2(const Real alpha, const VectorBase<OtherReal> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
  // remove __restrict__ if it causes compilation problems.
  register __restrict__  Real *data = data_;
  register __restrict__  OtherReal *other_data = v.data_;
  MatrixIndexT dim = dim_;
  if (alpha != 1.0)
    for (MatrixIndexT i = 0; i < dim; i++)
      data[i] += alpha*other_data[i]*other_data[i];
  else
    for (MatrixIndexT i = 0; i < dim; i++)
      data[i] += other_data[i]*other_data[i];
}

template
void VectorBase<float>::AddVec2(const float alpha, const VectorBase<double> &v);
template
void VectorBase<double>::AddVec2(const double alpha, const VectorBase<float> &v);


template<typename Real>
void VectorBase<Real>::Read(std::istream & is,  bool binary, bool add) {
  if (add) {
    Vector<Real> tmp(Dim());
    tmp.Read(is, binary, false);  // read without adding.
    if (this->Dim() != tmp.Dim()) {
      KALDI_ERR << "VectorBase::Read, size mismatch " << this->Dim()<<" vs. "<<tmp.Dim();
    }
    this->AddVec(1.0, tmp);
    return;
  } // now assume add == false.

  //  In order to avoid rewriting this, we just declare a Vector and
  // use it to read the data, then copy.
  Vector<Real> tmp;
  tmp.Read(is, binary, false);
  if (tmp.Dim() != Dim())
    KALDI_ERR << "VectorBase<Real>::Read, size mismatch "
              << Dim() << " vs. " << tmp.Dim();
  CopyFromVec(tmp);
}


template<typename Real>
void Vector<Real>::Read(std::istream & is,  bool binary, bool add) {
  if (add) {
    Vector<Real> tmp(this->Dim());
    tmp.Read(is, binary, false);  // read without adding.
    if (this->Dim() == 0) this->Resize(tmp.Dim());
    if (this->Dim() != tmp.Dim()) {
      KALDI_ERR << "Vector<Real>::Read, adding but dimensions mismatch "
                << this->Dim() << " vs. " << tmp.Dim();
    }
    this->AddVec(1.0, tmp);
    return;
  } // now assume add == false.

  std::ostringstream specific_error;
  MatrixIndexT pos_at_start = is.tellg();

  if (binary) {
    int peekval = Peek(is, binary);
    const char *my_token =  (sizeof(Real) == 4 ? "FV" : "DV");
    char other_token_start = (sizeof(Real) == 4 ? 'D' : 'F');
    if (peekval == other_token_start) {  // need to instantiate the other type to read it.
      typedef typename OtherReal<Real>::Real OtherType;  // if Real == float, OtherType == double, and vice versa.
      Vector<OtherType> other(this->Dim());
      other.Read(is, binary, false);  // add is false at this point.
      if (this->Dim() != other.Dim()) this->Resize(other.Dim());
      this->CopyFromVec(other);
      return;
    }
    std::string token;
    ReadToken(is, binary, &token);
    if (token != my_token) {
      specific_error << ": Expected token " << my_token << ", got " << token;
      goto bad;
    }
    int32 size;
    ReadBasicType(is, binary, &size);  // throws on error.
    if ((MatrixIndexT)size != this->Dim())  this->Resize(size);
    if (size > 0)
      is.read(reinterpret_cast<char*>(this->data_), sizeof(Real)*size);
    if (is.fail()) {
      specific_error << "Error reading vector data (binary mode); truncated "
          "stream? (size = " << size << ")";
      goto bad;
    }
    return;
  } else {  // Text mode reading; format is " [ 1.1 2.0 3.4 ]\n"
    std::string s;
    is >> s;
    // if ((s.compare("DV") == 0) || (s.compare("FV") == 0)) {  // Back compatibility.
    //  is >> s;  // get dimension
    //  is >> s;  // get "["
    // }
    if (is.fail()) { specific_error << "EOF while trying to read vector."; goto bad; }
    if (s.compare("[]") == 0) { Resize(0); return; } // tolerate this variant.
    if (s.compare("[")) { specific_error << "Expected \"[\" but got " << s; goto bad; }
    std::vector<Real> data;
    while (1) {
      int i = is.peek();
      if (i == '-' || (i >= '0' && i <= '9')) {  // common cases first.
        Real r;
        is >> r;
        if (is.fail()) { specific_error << "Failed to read number."; goto bad; }
        if (! std::isspace(is.peek()) && is.peek() != ']') {
          specific_error << "Expected whitespace after number."; goto bad;
        }
        data.push_back(r);
        // But don't eat whitespace... we want to check that it's not newlines
        // which would be valid only for a matrix.
      } else if (i == ' ' || i == '\t') {
        is.get();
      } else if (i == ']') {
        is.get();  // eat the ']'
        this->Resize(data.size());
        for (size_t j = 0; j < data.size(); j++)
          this->data_[j] = data[j];
        i = is.peek();
        if (static_cast<char>(i) == '\r') {
          is.get();
          is.get();  // get \r\n (must eat what we wrote)
        } else if (static_cast<char>(i) == '\n') { is.get(); } // get \n (must eat what we wrote)
        if (is.fail()) {
          KALDI_WARN << "After end of vector data, read error.";
          // we got the data we needed, so just warn for this error.
        }
        return;  // success.
      } else if (i == -1) {
        specific_error << "EOF while reading vector data.";
        goto bad;
      } else if (i == '\n' || i == '\r') {
        specific_error << "Newline found while reading vector (maybe it's a matrix?)";
        goto bad;
      } else {
        is >> s;  // read string.
        if (!KALDI_STRCASECMP(s.c_str(), "inf") ||
            !KALDI_STRCASECMP(s.c_str(), "infinity")) {
          data.push_back(std::numeric_limits<Real>::infinity());
          KALDI_WARN << "Reading infinite value into vector.";
        } else if (!KALDI_STRCASECMP(s.c_str(), "nan")) {
          data.push_back(std::numeric_limits<Real>::quiet_NaN());
          KALDI_WARN << "Reading NaN value into vector.";
        } else {
          specific_error << "Expecting numeric vector data, got " << s;
          goto  bad;
        }
      }
    }
  }
  // we never reach this line (the while loop returns directly).
bad:
  KALDI_ERR << "Failed to read vector from stream.  " << specific_error.str()
            << " File position at start is "
            << pos_at_start<<", currently "<<is.tellg();
}


template<typename Real>
void VectorBase<Real>::Write(std::ostream & os, bool binary) const {
  if (!os.good()) {
    KALDI_ERR << "Failed to write vector to stream: stream not good";
  }
  if (binary) {
    std::string my_token = (sizeof(Real) == 4 ? "FV" : "DV");
    WriteToken(os, binary, my_token);

    int32 size = Dim();  // make the size 32-bit on disk.
    KALDI_ASSERT(Dim() == (MatrixIndexT) size);
    WriteBasicType(os, binary, size);
    os.write(reinterpret_cast<const char*>(Data()), sizeof(Real) * size);
  } else {
    os << " [ ";
    for (MatrixIndexT i = 0; i < Dim(); i++)
      os << (*this)(i) << " ";
    os << "]\n";
  }
  if (!os.good())
    KALDI_ERR << "Failed to write vector to stream";
}


template<class Real>
void VectorBase<Real>::AddVec2(const Real alpha, const VectorBase<Real> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
  for (MatrixIndexT i = 0; i < dim_; i++) {
    data_[i] += v.data_[i]*v.data_[i]*alpha;
  }
}

// this <-- beta*this + alpha*M*v.
template<class Real>
void VectorBase<Real>::AddTpVec(const Real alpha, const TpMatrix<Real> &M,
                                const MatrixTransposeType trans,
                                const VectorBase<Real> &v,
                                const Real beta) {
  KALDI_ASSERT(dim_ == v.dim_ && dim_ == M.NumRows());
  if (beta == 0.0) {
    if (&v != this) CopyFromVec(v);
    MulTp(M, trans);
    if (alpha != 1.0) Scale(alpha);
  } else {
    Vector<Real> tmp(v);
    tmp.MulTp(M, trans);
    if (beta != 1.0) Scale(beta);  // *this <-- beta * *this
    AddVec(alpha, tmp);          // *this += alpha * M * v
  }
}

template<class Real>
Real VecMatVec(const VectorBase<Real> &v1, const MatrixBase<Real> &M,
               const VectorBase<Real> &v2) {
  KALDI_ASSERT(v1.Dim() == M.NumRows() && v2.Dim() == M.NumCols());
  Vector<Real> vtmp(M.NumRows());
  vtmp.AddMatVec(1.0, M, kNoTrans, v2, 0.0);
  return VecVec(v1, vtmp);
}

template
float VecMatVec(const VectorBase<float> &v1, const MatrixBase<float> &M,
                const VectorBase<float> &v2);
template
double VecMatVec(const VectorBase<double> &v1, const MatrixBase<double> &M,
                 const VectorBase<double> &v2);

template<class Real>
void Vector<Real>::Swap(Vector<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->dim_, other->dim_);
#ifdef KALDI_MEMALIGN_MANUAL
  std::swap(this->free_data_, other->free_data_);
#endif
}


template<>
void VectorBase<float>::AddDiagMat2(
    float alpha, const MatrixBase<float> &M,
    MatrixTransposeType trans, float beta) {
  if (trans == kNoTrans) {
    KALDI_ASSERT(this->dim_ == M.NumRows());
    MatrixIndexT rows = this->dim_, cols = M.NumCols(),
           mat_stride = M.Stride();
    float *data = this->data_;
    const float *mat_data = M.Data();
    for (MatrixIndexT i = 0; i < rows; i++, mat_data += mat_stride, data++)
      *data = beta * *data + alpha * cblas_sdot(cols, mat_data, 1, mat_data, 1);
  } else {
    KALDI_ASSERT(this->dim_ == M.NumCols());
    MatrixIndexT rows = M.NumRows(), cols = this->dim_,
           mat_stride = M.Stride();
    float *data = this->data_;
    const float *mat_data = M.Data();
    for (MatrixIndexT i = 0; i < cols; i++, mat_data++, data++)
      *data = beta * *data + alpha * cblas_sdot(rows, mat_data, mat_stride,
                                                mat_data, mat_stride);
  }
}

template<>
void VectorBase<double>::AddDiagMat2(
    double alpha, const MatrixBase<double> &M,
    MatrixTransposeType trans, double beta) {
  if (trans == kNoTrans) {
    KALDI_ASSERT(this->dim_ == M.NumRows());
    MatrixIndexT rows = this->dim_, cols = M.NumCols(),
           mat_stride = M.Stride();
    double *data = this->data_;
    const double *mat_data = M.Data();
    for (MatrixIndexT i = 0; i < rows; i++, mat_data += mat_stride, data++)
      *data = beta * *data + alpha * cblas_ddot(cols, mat_data, 1, mat_data, 1);
  } else {
    KALDI_ASSERT(this->dim_ == M.NumCols());
    MatrixIndexT rows = M.NumRows(), cols = this->dim_,
           mat_stride = M.Stride();
    double *data = this->data_;
    const double *mat_data = M.Data();
    for (MatrixIndexT i = 0; i < cols; i++, mat_data++, data++)
      *data = beta * *data + alpha * cblas_ddot(rows, mat_data, mat_stride,
                                                mat_data, mat_stride);
  }
}

template class Vector<float>;
template class Vector<double>;
template class VectorBase<float>;
template class VectorBase<double>;

}  // namespace kaldi



