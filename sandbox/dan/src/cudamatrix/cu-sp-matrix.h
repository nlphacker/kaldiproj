#ifndef KALDI_CUDAMATRIX_SP_MATRIX_H_
#define KALDI_CUDAMATRIX_SP_MATRIX_H_

#include <sstream>

#include "cudamatrix/cu-common.h"
#include "matrix/matrix-common.h"
#include "matrix/sp-matrix.h"
#include "cudamatrix/cu-stlvector.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-packed-matrix.h"

namespace kaldi {

template<typename Real>
class CuSpMatrix : public CuPackedMatrix<Real> {
 public:
  // friendships
  //friend class std:vector<CuMatrix<Real> >;
  
  // constructor
  CuSpMatrix(): CuPackedMatrix<Real>() {}
  
  explicit CuSpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
    : CuPackedMatrix<Real>(r, resize_type) {}

  explicit CuSpMatrix(const SpMatrix<Real> &orig)
    : PackedMatrix<Real>(orig) {}

  explicit CuSpMatrix(const CuSpMatrix<Real> &orig)
    : CuPackedMatrix<Real>(orig) {}

  // deconstructor
  ~CuSpMatrix() {}

  // resize
  inline void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero) {
    CuPackedMatrix<Real>::Resize(nRows, resize_type);
  }

  // copyfromsp
  void CopyFromSp(const CuSpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }
  void CopyFromSp(const SpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }

  // operators
  //inline Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    
  private:
  
  };

} // namespace

#endif
