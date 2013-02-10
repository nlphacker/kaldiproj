// cudamatrix/cu-math.cc

// Copyright 2009-2012  Karel Vesely
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

#include "util/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"

namespace kaldi {

namespace cu {

/*
 * templated functions wrapping the ANSI-C CUDA kernel functions 
 */


template<typename Real>
void RegularizeL1(CuMatrixBase<Real> *weight, CuMatrixBase<Real> *grad, Real l1, Real lr) {
  KALDI_ASSERT(SameDim(*weight, *grad));
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(weight->NumCols(), CUBLOCK), n_blocks(weight->NumRows(), CUBLOCK));

    cuda_regularize_l1(dimGrid, dimBlock, weight->data_, grad->data_, l1, lr, weight->Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<Real> &weight2 = weight->Mat();
    MatrixBase<Real> &grad2 = grad->Mat();
    for(MatrixIndexT r=0; r<weight2.NumRows(); r++) {
      for(MatrixIndexT c=0; c<weight2.NumCols(); c++) {
        
        if(weight2(r,c)==0.0) continue; // skip L1 if zero weightght!

        Real l1_signed = l1;
        if (weight2(r, c) < 0.0) 
          l1_signed = -l1;

        Real before = weight2(r, c);
        Real after = weight2(r, c) - lr*grad2(r, c) - l1_signed;
        if ((after > 0.0) ^ (before > 0.0)) {
          weight2(r, c) = 0.0;
          grad2(r, c) = 0.0;
        } else {
          weight2(r, c) -= l1_signed;
        }
      }
    }
  }
}


template<typename Real>
void Randomize(const CuMatrixBase<Real> &src,
               const CuStlVector<int32> &copy_from_idx,
               CuMatrixBase<Real> *tgt) {

  KALDI_ASSERT(src.NumCols() == tgt->NumCols());
  KALDI_ASSERT(src.NumRows() == tgt->NumRows());
  KALDI_ASSERT(copy_from_idx.Dim() <= tgt->NumRows());

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CUBLOCK), n_blocks(copy_from_idx.Dim(), CUBLOCK));
    
    MatrixDim dimsrc = src.Dim(); dimsrc.rows=copy_from_idx.Dim();
    MatrixDim dimtgt = tgt->Dim(); dimtgt.rows=copy_from_idx.Dim();

    cuda_randomize(dimGrid, dimBlock, tgt->data_, src.data_, copy_from_idx.Data(), dimtgt, dimsrc);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    // randomize in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const std::vector<int32> &copy_from_idxvec = copy_from_idx.Vec();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    for(int32 i=0; i<copy_from_idx.Dim(); i++) {
      tgtmat.Row(i).CopyFromVec(srcmat.Row(copy_from_idxvec[i]));
    }
  }
} 



template<typename Real>
void Splice(const CuMatrix<Real> &src, const CuStlVector<int32> &frame_offsets, CuMatrix<Real> *tgt) {

  KALDI_ASSERT(src.NumCols()*frame_offsets.Dim() == tgt->NumCols());
  KALDI_ASSERT(src.NumRows() == tgt->NumRows());

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CUBLOCK), n_blocks(tgt->NumRows(), CUBLOCK));
    
    cuda_splice(dimGrid, dimBlock, tgt->data_, src.data_, frame_offsets.Data(), tgt->Dim(), src.Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    // expand in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const std::vector<int32> &frame_offsetvec = frame_offsets.Vec();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    //
    for(int32 r=0; r < tgtmat.NumRows(); r++) {
      for(int32 off=0; off < static_cast<int32>(frame_offsetvec.size()); off++) {
        int32 r_off = r + frame_offsetvec[off];
        if(r_off < 0) r_off = 0;
        if(r_off >= srcmat.NumRows()) r_off = srcmat.NumRows()-1;
        memcpy(tgtmat.RowData(r)+off*srcmat.NumCols(),srcmat.RowData(r_off),sizeof(Real)*srcmat.NumCols());
      }
    }
  }
}



template<typename Real>
void Copy(const CuMatrix<Real> &src, const CuStlVector<int32> &copy_from_indices, CuMatrix<Real> *tgt) { 

  KALDI_ASSERT(copy_from_indices.Dim() == tgt->NumCols());
  KALDI_ASSERT(src.NumRows() == tgt->NumRows());

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(tgt->NumCols(), CUBLOCK), n_blocks(tgt->NumRows(), CUBLOCK));
    
    cuda_copy(dimGrid, dimBlock, tgt->data_, src.data_, copy_from_indices.Data(), tgt->Dim(), src.Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    // expand in CPU
    const MatrixBase<Real> &srcmat = src.Mat();
    const std::vector<int32> &copy_from_indicesvec = copy_from_indices.Vec();
    MatrixBase<Real> &tgtmat = tgt->Mat();
    //
    for(int32 r=0; r < tgtmat.NumRows(); r++) {
      for(int32 c=0; c < static_cast<int32>(copy_from_indicesvec.size()); c++) {
        tgtmat(r,c) = srcmat(r,copy_from_indicesvec[c]);
      }
    }
  }
}

// instantiate the templates.
template
void RegularizeL1(CuMatrixBase<float> *weight, CuMatrixBase<float> *grad, float l1, float lr);
template
void RegularizeL1(CuMatrixBase<double> *weight, CuMatrixBase<double> *grad, double l1, double lr);

template
void Splice(const CuMatrix<float> &src, const CuStlVector<int32> &frame_offsets, CuMatrix<float> *tgt);
template
void Splice(const CuMatrix<double> &src, const CuStlVector<int32> &frame_offsets, CuMatrix<double> *tgt);
template
void Copy(const CuMatrix<float> &src, const CuStlVector<int32> &copy_from_indices, CuMatrix<float> *tgt);
template
void Copy(const CuMatrix<double> &src, const CuStlVector<int32> &copy_from_indices, CuMatrix<double> *tgt);

template
void Randomize(const CuMatrixBase<float> &src,
               const CuStlVector<int32> &copy_from_idx,
               CuMatrixBase<float> *tgt);
template
void Randomize(const CuMatrixBase<double> &src,
               const CuStlVector<int32> &copy_from_idx,
               CuMatrixBase<double> *tgt);



} //namespace cu

} //namespace kaldi

