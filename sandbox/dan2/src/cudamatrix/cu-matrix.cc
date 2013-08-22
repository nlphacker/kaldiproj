// cudamatrix/cu-matrix.cc

// Copyright 2009-2012  Karel Vesely
//                      Lucas Ondel
//                      Johns Hopkins University (author: Daniel Povey)

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


#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"
#include "cu-kernels.h"
#include "cu-randkernels.h"
#include "cu-rand-inl.h"
#include "cu-choleskykernels.h"
#include "cu-stlvector.h"
#include "cu-math.h"

namespace kaldi {

template<typename Real>
void CuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols,
                            MatrixResizeType resize_type) {
  // This code does not currently support the other resize_type options.
  KALDI_ASSERT(resize_type == kSetZero || resize_type == kUndefined);
  if (rows * cols == 0) KALDI_ASSERT(rows == 0 && cols == 0);
  if (this->num_rows_ == rows && this->num_cols_ == cols) {
    if (resize_type == kSetZero) this->SetZero();
    return;
  }

  if (this->num_rows_ != 0)
    this->Destroy();
  if (rows == 0) return;  
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    MatrixIndexT row_bytes = cols * sizeof(Real);
    size_t pitch;
    CU_SAFE_CALL(cudaMallocPitch(reinterpret_cast<void**>(&this->data_), &pitch,
                               row_bytes, rows));
    this->num_rows_ = rows;
    this->num_cols_ = cols; 
    this->stride_ = pitch / sizeof(Real);
    if (resize_type == kSetZero) this->SetZero();
  } else
#endif
  { // Let the initializer of Matrix<Real> handle the allocation,
    // and then just do Swap which will switch the pointers.
    // This wastes a few instructions but is simple to code.
    Matrix<Real> mat(rows, cols, resize_type);
    this->Swap(&mat);
  }
}

template<typename Real>
void CuMatrix<Real>::Destroy() {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    if (this->data_ != NULL) {
      CU_SAFE_CALL(cudaFree(this->data_));
    }
  } else
  #endif
  {
    if (this->data_ != NULL) KALDI_MEMALIGN_FREE(this->data_);
  }
  this->data_ = NULL;
  this->num_rows_ = 0;
  this->num_cols_ = 0;
  this->stride_ = 0;
}

template<typename Real>
void CuMatrix<Real>::Swap(CuMatrix<Real> *mat) {
  std::swap(mat->data_, this->data_);
  std::swap(mat->num_cols_, this->num_cols_);
  std::swap(mat->num_rows_, this->num_rows_);
  std::swap(mat->stride_, this->stride_);
}


template<typename Real>
void CuMatrix<Real>::Swap(Matrix<Real> *mat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) {
      if (mat->num_rows_ != 0) {
        // *this is empty, but mat is nonempty.
        this->Resize(mat->num_rows_, mat->num_cols_, kUndefined);
        this->CopyFromMat(*mat);
        mat->Resize(0, 0);
      }
      // else both are empty.
    } else { // *this is nonempty.
      if (mat->num_rows_ != 0) {
        // Both *this and *mat are nonempty.  Recurse to simpler cases.
        // this could be done more efficiently in the case where
        // the size does not change.
        Matrix<Real> temp;
        this->Swap(&temp); // now temp is full, *this is empty.
        mat->Swap(&temp); // now mat has data from *this, temp has
        // data from mat.
        this->Swap(&temp); // copy data in mat to *this, which is now empty.
      } else { // *this is full but *mat is empty.
        mat->Resize(this->num_rows_, this->num_cols_, kUndefined);
        this->CopyToMat(mat);
        this->Destroy();
      }
    }
  } else
#endif
  {
    std::swap(mat->data_, this->data_);
    std::swap(mat->num_cols_, this->num_cols_);
    std::swap(mat->num_rows_, this->num_rows_);
    std::swap(mat->stride_, this->stride_);
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyFromMat(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_ * sizeof(Real);
    MatrixIndexT src_pitch = src.Stride() * sizeof(Real);
    MatrixIndexT width = src.NumCols() * sizeof(Real);
    CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, src.data_, src_pitch,
                            width, src.num_rows_, cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatD2D",tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(src.Mat());
  }
}

template<>
template<>
void CuMatrixBase<double>::CopyFromMat(const CuMatrixBase<float> &M,
                                     MatrixTransposeType Trans) {
  //KALDI_ASSERT(sizeof(Real) != sizeof(OtherReal));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(), CUBLOCK), n_blocks(M.NumRows(), CUBLOCK));
    if (Trans == kNoTrans) {
      KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());
      cuda_copy_from_mat_df(dimGrid, dimBlock, data_, M.data_,
                            Dim(), M.Dim());
    } else {
      KALDI_ASSERT(num_rows_ == M.NumCols() && num_cols_ == M.NumRows ());
      cuda_copy_from_mat_df_trans(dimGrid, dimBlock, data_, M.data_, Dim(), M.Dim());
    }
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}

template<>
template<>
void CuMatrixBase<float>::CopyFromMat(const CuMatrixBase<double> &M,
                                      MatrixTransposeType Trans) {
  //KALDI_ASSERT(sizeof(Real) != sizeof(OtherReal));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(), CUBLOCK), n_blocks(M.NumRows(), CUBLOCK));
    if (Trans == kNoTrans) {
      KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());

      cuda_copy_from_mat_fd(dimGrid, dimBlock, data_, M.data_,
                            Dim(), M.Dim());
    } else {

      KALDI_ASSERT(num_rows_ == M.NumCols() && num_cols_ == M.NumRows ());
      cuda_copy_from_mat_fd_trans(dimGrid, dimBlock, data_, M.RowData(0),
                                    Dim(), M.Dim());
    }
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}

template<typename Real>
template<typename OtherReal>
void CuMatrixBase<Real>::CopyFromTp(const CuTpMatrix<OtherReal> &M,
                                    MatrixTransposeType Trans) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimGrid = 1;
    int dimBlock = num_rows_;
    SetZero();
    if (Trans == kNoTrans) {
      cuda_copy_from_tp(dimGrid, dimBlock, data_, M.Data(), Dim());
    } else {
      cuda_copy_from_tp_trans(dimGrid, dimBlock, data_, M.Data(), Dim());      
    }
  } else
#endif
  {
    Mat().CopyFromTp(M.Mat(), Trans);
  }
}
/*
// template instantiations.
template
void CuMatrixBase<float>::CopyFromMat(const CuMatrixBase<double> & M,
                                      MatrixTransposeType Trans);
template
void CuMatrixBase<double>::CopyFromMat(const CuMatrixBase<float> & M,
                                       MatrixTransposeType Trans);

template
void CuMatrixBase<float>::CopyFromMat(const CuMatrixBase<float> & M,
                                      MatrixTransposeType Trans);
template
void CuMatrixBase<double>::CopyFromMat(const CuMatrixBase<double> & M,
                                       MatrixTransposeType Trans);
*/

template<typename Real>
void CuMatrixBase<Real>::CopyFromMat(const MatrixBase<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);
    CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch,
                            width, src.NumRows(), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatH2D",tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(src);
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyFromSp(const CuSpMatrix<Real> &M) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(NumRows(),CUBLOCK));
    cuda_copy_from_sp(dimGrid, dimBlock, M.Data(), data_, num_rows_, Dim());
    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromSp",tim.Elapsed());
  } else
#endif
  {
    //Mat().CopyFromSp(M.Mat());
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyToMat(MatrixBase<Real> *dst) const {
  KALDI_ASSERT(dst->NumRows() == NumRows() && dst->NumCols() == NumCols());
  
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 

    Timer tim;
   
    MatrixIndexT src_pitch = stride_*sizeof(Real);
    MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
    MatrixIndexT width = NumCols()*sizeof(Real);
    CU_SAFE_CALL(cudaMemcpy2D(dst->data_, dst_pitch, this->data_, src_pitch,
                            width, this->num_rows_, cudaMemcpyDeviceToHost));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
  } else
  #endif
  {
    dst->CopyFromMat(Mat());
  }
}


/*
template<typename Real>
void CuMatrixBase<Real>::CopyRowsFromMat(int32 r, const CuMatrixBase<Real> &src, int32 src_ro, int32 dst_ro) {
  KALDI_ASSERT(r+src_ro <= src.NumRows());
  KALDI_ASSERT(r+dst_ro <= NumRows());
  KALDI_ASSERT(NumCols() == src.NumCols());
   
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);

    const Real *p_src = src.Data() + src_ro*src.Stride();  
    Real *p_dst = data_ + dst_ro*stride_;

    CU_SAFE_CALL(cudaMemcpy2D(p_dst, dst_pitch, p_src, src_pitch, width, r, cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyRowsD2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(Data()+dst_ro*stride_, src.Data()+src_ro*src.Stride(), r*stride_*sizeof(Real));
  }
} */



template<typename Real>
void CuMatrix<Real>::Read(std::istream &is, bool binary) {
  Matrix<Real> temp;
  temp.Read(is, binary);
  Destroy();
  Swap(&temp);
}

template<typename Real>
void CuMatrix<Real>::Write(std::ostream &os, bool binary) const {
  Matrix<Real> temp(this->num_rows_, this->num_cols_, kUndefined);
  this->CopyToMat(&temp);
  temp.Write(os, binary); 
}

template<typename Real>
void CuMatrixBase<Real>::SetZero() {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    CU_SAFE_CALL(cudaMemset(data_, 0, num_rows_*stride_*sizeof(Real)));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero", tim.Elapsed());
  } else
  #endif
  {
    Mat().SetZero();
  }
}




/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real> 
void CuMatrixBase<Real>::Set(Real value) {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_set_const(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Set(value);
  }
}

// set zero the upper diagonal
// no cpu implementation yet. Check with Dan.
template<typename Real>
void CuMatrixBase<Real>::SetZeroUpperDiag() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_set_zero_above_diag(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real> 
void CuMatrixBase<Real>::Add(Real value) { 
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Add(value);
  }
}


template<typename Real> 
void CuMatrixBase<Real>::Scale(Real value) { 
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_scale(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(value);
  }
}



template<typename Real> 
void CuMatrixBase<Real>::ApplyLog() { 
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_apply_log(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().ApplyLog();
  }
}



template<typename Real>
void CuMatrixBase<Real>::MulElements(const CuMatrixBase<Real>& A) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());
    KALDI_ASSERT(stride_ == A.Stride());
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_elements(dimGrid, dimBlock, data_, A.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().MulElements(A.Mat());
  }
}



template<typename Real>
void CuMatrixBase<Real>::MulColsVec(const CuVectorBase<Real> &scale) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_cols_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().MulColsVec(scale.Vec());
  }
}



template<typename Real>
void CuMatrixBase<Real>::MulRowsVec(const CuVectorBase<Real> &scale) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_rows_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    Mat().MulRowsVec(scale.Vec());
  }
}



template<typename Real>
void CuMatrixBase<Real>::DivRowsVec(const CuVectorBase<Real> &div) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(div.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_div_rows_vec(dimGrid, dimBlock, data_, div.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
#endif
  {
    Vector<Real> temp(div.Vec()); // will copy.
    temp.InvertElements();
    Mat().MulRowsVec(temp);
  }
}



template<typename Real>
void CuMatrixBase<Real>::AddMat(Real alpha, const CuMatrixBase<Real>& A, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(A.NumRows() == NumRows());
    KALDI_ASSERT(A.NumCols() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_mat(dimGrid, dimBlock, alpha, A.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(beta);
    Mat().AddMat(alpha, A.Mat());
  }
}



template<typename Real>
void CuMatrixBase<Real>::AddVecToCols(Real alpha,
                                      const CuVectorBase<Real> &col,
                                      Real beta) { 
  if (col.Dim() != NumRows()) {
    KALDI_ERR << "Non matching dimensions: Rows:" << NumRows() << " VectorDim:" << col.Dim();
  }

  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_vec_to_cols(dimGrid, dimBlock, alpha, col.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    if (beta != 1.0) Mat().Scale(beta);
    Mat().AddVecToCols(alpha, col.Vec());
  }
}



template<typename Real>
void CuMatrixBase<Real>::AddVecToRows(Real alpha,
                                      const CuVectorBase<Real> &row,
                                      Real beta) { 
  if (row.Dim() != NumCols()) {
    KALDI_ERR << "Non matching dimensions: Cols:" << NumCols() << " VectorDim:" << row.Dim();
  }
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_vec_to_rows(dimGrid, dimBlock, alpha, row.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    if (beta != 1.0) Mat().Scale(beta);
    Mat().AddVecToRows(alpha, row.Vec());
  }
}



/**
 * C++ templated wrapper of ANSI-C CUBLAS function GEMM (matrix multiply)
 */
#if HAVE_CUDA == 1
template<typename Real> inline void cublas_gemm(char transa, char transb, int m, int n,int k, Real alpha, const Real *A, int lda,const Real *B, int ldb, Real beta, Real *C, int ldc) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_gemm<float>(char transa, char transb, int m, int n,int k, float alpha, const float *A, int lda,const float *B, int ldb, float beta, float *C, int ldc) {
  cublasSgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
template<> inline void cublas_gemm<double>(char transa, char transb, int m, int n,int k, double alpha, const double *A, int lda,const double *B, int ldb, double beta, double *C, int ldc) {
  cublasDgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
#endif



/*
 * Method wrapping the CUBLAS function GEMM
 */
template<typename Real>
void CuMatrixBase<Real>::AddMatMat(
    Real alpha, const CuMatrixBase<Real>& A, MatrixTransposeType transA,
    const CuMatrixBase<Real>& B, MatrixTransposeType transB, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
    // keep trans..., just swap A&B matrices: A->B B->A
    MatrixIndexT m = ((transB==kTrans)? B.NumRows() : B.NumCols()); 
    MatrixIndexT n = ((transA==kTrans)? A.NumCols() : A.NumRows());
    MatrixIndexT k = ((transB==kTrans)? B.NumCols() : B.NumRows());
    MatrixIndexT k1 = ((transA==kTrans)? A.NumRows() : A.NumCols());

    KALDI_ASSERT(m == NumCols());
    KALDI_ASSERT(n == NumRows());
    KALDI_ASSERT(k == k1);

    Timer tim;

    cublas_gemm((transB==kTrans?'T':'N'), (transA==kTrans?'T':'N'), m, n, k, 
                alpha, B.data_, B.Stride(), A.data_, A.Stride(), 
                beta, data_, Stride());

    CU_SAFE_CALL(cublasGetError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().AddMatMat(alpha, A.Mat(), transA, B.Mat(), transB, beta);
  }
}

/// Element-wise, does (*this) += alpha + A * B / C.
/// In the special case that C == 0, adds nothing.
/// we have implemented the first version which all trans* = kNoTrans
template<typename Real>
void CuMatrixBase<Real>::AddMatMatDivMatElements(
    Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA,
    const CuMatrixBase<Real> &B, MatrixTransposeType transB,
    const CuMatrixBase<Real> &C, MatrixTransposeType transC,
    Real beta) {
  KALDI_ASSERT(num_rows_ == num_cols_ &&
               A.NumRows() == num_rows_ && A.NumCols() == num_cols_ &&
               B.NumRows() == num_rows_ && B.NumCols() == num_cols_ &&
               C.NumRows() == num_rows_ && C.NumCols() == num_cols_);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CUBLOCK), n_blocks(num_rows_, CUBLOCK));
    cuda_ammdm_elements(dimGrid, dimBlock, alpha, data_, A.RowData(0),
                        B.RowData(0), C.RowData(0), beta, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
  }
}
template<typename Real>
void CuMatrixBase<Real>::Sigmoid(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CUBLOCK), n_blocks(src.NumRows(), CUBLOCK));

    cuda_sigmoid(dimGrid, dimBlock, this->data_, src.data_, src.Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Sigmoid(src.Mat());
  }
}


template<typename Real> // Y->this, X->src
void CuMatrixBase<Real>::ApplySoftMax(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

#if 1
    // enable 'tree-reduce' functions, 
    //find maximum in each row (tree reduction)
    CuStlVector<int32> max_id;
    src.FindRowMaxId(&max_id); 
    //in each row subtract maximum, apply exp (grid kernel)
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(src.num_cols_, CUBLOCK), n_blocks(src.num_rows_, CUBLOCK));
    cuda_softmax_part(dimGrid, dimBlock, src.data_, max_id.Data(), this->data_, src.Dim()); 
    //sum the rows to get normalizers (tree reduction) 
    CuVector<Real> sum(src.num_rows_);
    sum.AddColSumMat(1.0, *this, 0.0);
    //divide by normalizers to get posteriors (grid kernel)
    this->DivRowsVec(sum);
#else
    // disable 'tree-reduce' functions, 
    // slower, but can be used for debugging
    size_t dimBlock = CUBLOCK;
    size_t dimGrid  = n_blocks(src.num_rows_, CUBLOCK);

    cuda_softmax(dimGrid, dimBlock, data_, src.data_, src.Dim());
    CU_SAFE_CALL(cudaGetLastError());
#endif

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<Real> &mat(this->Mat());
    mat.CopyFromMat(src.Mat());
    for(MatrixIndexT r = 0; r < mat.NumRows(); r++) {
      mat.Row(r).ApplySoftMax();
    }
  }
}

// DiffSigmoid(Ein, Y, Eout) -> Eout.DiffSigmoid(Y, Ein).
template<typename Real> // Eout -> *this, Ein -> diff, Y -> value
void CuMatrixBase<Real>::DiffSigmoid(const CuMatrixBase<Real> &value,
                                     const CuMatrixBase<Real> &diff) {
  KALDI_ASSERT(SameDimAndStride(*this, value) && SameDimAndStride(*this, diff));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CUBLOCK), n_blocks(num_rows_, CUBLOCK));

    cuda_diff_sigmoid(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().DiffSigmoid(value.Mat(), diff.Mat());
  }
}

  
template<typename Real>
void CuMatrixBase<Real>::Tanh(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CUBLOCK), n_blocks(src.NumRows(), CUBLOCK));

    cuda_tanh(dimGrid, dimBlock, this->data_, src.data_, src.Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Tanh(src.Mat());
  }
}



template<typename Real> // Ein -> diff, Y -> value
void CuMatrixBase<Real>::DiffTanh(const CuMatrixBase<Real> &value,
                                  const CuMatrixBase<Real> &diff) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CUBLOCK), n_blocks(num_rows_, CUBLOCK));

    cuda_diff_tanh(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().DiffTanh(value.Mat(), diff.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::FindRowMaxId(CuStlVector<int32> *id) const {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
     
    // initialize the vectors
    CuVector<Real> max(num_rows_);
    max.Set(-1e21);
    id->Resize(num_rows_);
    id->Set(-1);

    MatrixDim d=Dim();// only stride will be used!
   
    // process per 256 column blocks 
    for(int32 block=0; (block+1)*256 <= num_cols_; block++) {
      dim3 dimBlock(256, 1);
      dim3 dimGrid(1, num_rows_);
      int32 offset=block*256;

      cuda_find_row_max_id(dimGrid, dimBlock, data_ + offset,
                           max.data_, id->Data(), offset, d);
    }
    
    // process the remainder
    int32 div = num_cols_ / 256;
    int32 mod = num_cols_ % 256;
    if (mod != 0) {
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, num_rows_);
      int32 offset=div*256;
      
      cuda_find_row_max_id(dimGrid, dimBlock, data_ + offset,
                           max.data_, id->Data(), offset, d);
    }
    // now we have the indices!
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    // allocate index buffer
    id->Resize(num_rows_);
    id->Set(-1);
    // find maxima
    MatrixIndexT num_rows = num_rows_, num_cols = num_cols_;
    for(MatrixIndexT r = 0; r < num_rows; r++) {
      Real max = -1e21;
      int32 max_id = -1;
      const Real *row_data = Mat().RowData(r);
      for(MatrixIndexT c = 0; c < num_cols; c++) {
        if (max < row_data[c]) {
          max = row_data[c];
          max_id = c;
        }
      }
      id->Vec()[r] = max_id;
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::DiffXent(const CuStlVector<int32> &tgt,
                                  CuVector<Real> *log_post_tgt) {
  
  KALDI_ASSERT(tgt.Dim() == num_rows_);
  log_post_tgt->Resize(tgt.Dim());

#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(1, CUBLOCK*8);
    dim3 dimGrid(1, n_blocks(tgt.Dim(), CUBLOCK*8));
    cuda_diff_xent(dimGrid, dimBlock, tgt.Data(), data_,
                   log_post_tgt->data_, Dim());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    MatrixIndexT num_rows = num_rows_;
    for(int32 r = 0; r < num_rows; r++) {
      int32 col_tgt = tgt.Vec()[r];
      Real &value = Mat()(r, col_tgt);
      log_post_tgt->Vec()(r) = log(value);
      value -= 1.0;
    }
  }
}

// Cholesky method may be only called for symmetric matrices.
template<typename Real>
void CuMatrixBase<Real>::Cholesky() {
  KALDI_ASSERT(this->NumRows() == this->NumCols());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int TILE_SIZE = 16;
    int n_blocks = (num_rows_ + TILE_SIZE - 1) / TILE_SIZE;

    dim3 threads(TILE_SIZE,TILE_SIZE);
    dim3 logrid;
     
    for (int i = n_blocks; i > 2; i--) {
      cuda_factorize_diagonal_block(data_, n_blocks-i, Dim());
      cudaThreadSynchronize();

      cuda_strip_update(data_, n_blocks-i, i, Dim());
      cudaThreadSynchronize();
      
      cuda_diag_update(data_, n_blocks-i, i, Dim());
      cudaThreadSynchronize();
      
      cuda_lo_update(data_, n_blocks-i, n_blocks, i, Dim());
      cudaThreadSynchronize();      
    }
    
    if (n_blocks > 1) {
      cuda_factorize_diagonal_block(data_, n_blocks-2, Dim());
      cudaThreadSynchronize();
      
      cuda_strip_update(data_, n_blocks-2, 2, Dim());
      cudaThreadSynchronize();
      
      cuda_diag_update(data_, n_blocks-2, 2, Dim());
      cudaThreadSynchronize();
      
    }

    
    cuda_factorize_diagonal_block(data_, n_blocks-1, Dim());
    cudaThreadSynchronize();

    // set the upper diagonal equal to zero
    this->SetZeroUpperDiag();
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    
  } else
#endif
  {
    SpMatrix<Real> sp(this->NumRows(), kUndefined);
    sp.CopyFromMat(this->Mat(), kTakeLower);
    TpMatrix<Real> tp(this->NumRows());
    tp.Cholesky(sp);
    this->Mat().CopyFromTp(tp);
  }
}

#if HAVE_CUDA
template<typename Real> inline void cublas_trsm(int m, int n, Real alpha, const Real* A, int lda, Real* B, int ldb) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_trsm<float>(int m, int n, float alpha, const float* A, int lda, float* B, int ldb) {
  cublasStrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
template<> inline void cublas_trsm<double>(int m, int n, double alpha, const double* A, int lda, double* B, int ldb) {
  cublasDtrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
#endif

template<typename Real>
void CuMatrixBase<Real>::InvertPSD() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
 
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(NumRows(),CUBLOCK));
    CuMatrix<Real> temp(num_rows_,num_rows_);
    int dim = num_rows_;
    Real value = 1.0;
    cuda_set_diag(dimGrid, dimBlock, temp.RowData(0), value, temp.Dim());
    Matrix<Real> A(dim,dim);
    temp.CopyToMat(&A);
    this->Cholesky();
    //CuSpMatrix<Real> L(*this, kTakeLower);
    Real alpha = 1.0;
    cublas_trsm(num_rows_,num_rows_,alpha,data_,stride_,temp.RowData(0),temp.Dim().stride);
    
    //CuSpMatrix<Real> L(temp, kTakeLower);
    //CuMatrix<Real> L1(dim,dim);
    //L1.CopyFromSp(L);
    //L1.SetZeroUpperDiag();
    Matrix<Real> L_test(dim,dim);
    temp.CopyToMat(&L_test);
    this->AddMatMat(1,temp,kTrans,temp,kNoTrans,0);
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    this->Mat().Invert(); // This is inefficient as we don't make
    // use of the fact that we're symmetric, but anyway if we're not
    // using CUDA this function typically shouldn't be called, because its
    // only envisaged usage is to be call from the CUDA version of
    // CuSpMatrix::Invert().
  }
}


template<class Real>
Real TraceMatMat(const CuMatrixBase<Real> &A,
                 const CuMatrixBase<Real> &B,
                 MatrixTransposeType trans) {
  Real result = 0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(A.NumRows(), CUBLOCK));
    Real* device_result;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(Real)));
    CU_SAFE_CALL(cudaMemset(device_result, 0, sizeof(Real)));
    if (trans == kNoTrans) {
      KALDI_ASSERT(A.NumRows() == B.NumCols() && A.NumCols() == B.NumRows());
      cuda_trace_mat_mat(dimGrid, dimBlock, A.RowData(0), B.RowData(0), A.Dim(), B.Dim(), device_result);
    } else {
      KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
      cuda_trace_mat_mat_trans(dimGrid, dimBlock, A.RowData(0), B.RowData(0), A.Dim(), B.Dim(), device_result);
    }
    CU_SAFE_CALL(cudaGetLastError());
    CU_SAFE_CALL(cudaMemcpy(&result, device_result, sizeof(Real), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    result = TraceMatMat(A.Mat(), B.Mat(), trans);
  }
  return result;
}

template
float TraceMatMat(const CuMatrixBase<float> &A,
                  const CuMatrixBase<float> &B,
                  MatrixTransposeType trans);
template
double TraceMatMat(const CuMatrixBase<double> &A,
                   const CuMatrixBase<double> &B,
                   MatrixTransposeType trans);


template<typename Real>
void CuMatrixBase<Real>::CopyRowsFromVec(const CuVectorBase<Real> &rv) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    if (rv.Dim() == num_rows_*num_cols_) {
      if (stride_ == num_cols_) {
        const Real* rv_data = rv.Data();
        cudaMemcpy(data_, rv_data, sizeof(Real)*num_rows_*num_cols_, cudaMemcpyDeviceToDevice);
      } else {
        const Real *rv_data = rv.Data();
        for (MatrixIndexT r = 0; r < num_rows_; r++) {
          Real *row_data = RowData(r);
          cudaMemcpy(row_data, rv_data, sizeof(Real)*num_cols_, cudaMemcpyDeviceToDevice);
          rv_data += num_cols_;
        }
      }
    } else if (rv.Dim() == num_cols_) {
      const Real *rv_data = rv.Data();
      for (MatrixIndexT r = 0; r < num_rows_; r++)
        cudaMemcpy(RowData(r), rv_data, sizeof(Real)*num_cols_, cudaMemcpyDeviceToDevice);
    } else {
      KALDI_ERR << "Wrong sized arguments";
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().CopyRowsFromVec(rv.Vec());
  }

}

template<typename Real>
void CuMatrixBase<Real>::CopyColFromVec(const CuVectorBase<Real> &rv,
                                        const MatrixIndexT col) {
  KALDI_ASSERT(rv.Dim() == num_rows_ &&
               static_cast<UnsignedMatrixIndexT>(col) <
               static_cast<UnsignedMatrixIndexT>(num_cols_));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(NumRows(), CUBLOCK));
    cuda_copy_col_from_vec(dimGrid, dimBlock, data_, rv.Data(), col, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().CopyColFromVec(rv.Vec(), col);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyPow(Real power) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_apply_pow(dimGrid, dimBlock, data_, power, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyPow(power);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyExp() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_apply_exp(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyExp();
  }
}


template<typename Real>
void CuMatrixBase<Real>::ApplyFloor(Real floor_val) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_apply_floor(dimGrid, dimBlock, data_, floor_val, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyFloor(floor_val);
  }
}

/*
template<typename Real>
Real CuMatrixBase<Real>::Sum() const {
  Real result = 0.0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));
    Real* device_result;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(Real)));
    CU_SAFE_CALL(cudaMemset(device_result,0, sizeof(Real)));
    cuda_sum(dimGrid, dimBlock, data_, device_result, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CU_SAFE_CALL(cudaMemcpy(&result, device_result, sizeof(Real), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    result = Mat().Sum();
  }
  return result;
}
*/

template<typename Real>
void CuMatrix<Real>::SetRandn() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuRand<Real> tmp;
    tmp.RandGaussian(this);
  }
#endif
}

/// Copy constructor from another type.
template<typename Real>
template<typename OtherReal>
CuMatrix<Real>::CuMatrix(const CuMatrixBase<OtherReal> & M,
                         MatrixTransposeType trans) : CuMatrixBase<Real>() {

  if (trans == kNoTrans) {
    Resize(M.NumRows(), M.NumCols());
    this->CopyFromMat(M);
  } else {
    Resize(M.NumCols(), M.NumRows());
    this->CopyFromMat(M, kTrans);
  }

}

// Instantiate this constructor for float->double and double->float.
template
CuMatrix<float>::CuMatrix(const CuMatrixBase<double> & M,
                          MatrixTransposeType trans);
template
CuMatrix<double>::CuMatrix(const CuMatrixBase<float> & M,
                           MatrixTransposeType trans);

/*
template<typename Real>
CuMatrix<Real>::DeriveLastLayerComponent(int32 i, int32 label,
                                         Real weight, Real this_prob) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    cuda_derive_last_layer_component(i, label, weight, this_prob);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
  {

  }
}
*/

/**
 * Print the matrix to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrixBase<Real> &mat) {
  Matrix<Real> temp(mat.NumRows(), mat.NumCols());
  mat.CopyToMat(&temp);
  out << temp;
  return out;
}

// instantiate the template
template
std::ostream &operator << (std::ostream &out, const CuMatrixBase<float> &mat);
template 
std::ostream &operator << (std::ostream &out, const CuMatrixBase<double> &mat);


// Instantiate classes CuMatrix and CuMatrixBase for float and double.
template class CuMatrix<float>;
template class CuMatrix<double>;
template class CuMatrixBase<float>;
template class CuMatrixBase<double>;


} // namespace kaldi