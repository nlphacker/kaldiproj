// cudamatrix/cu-matrix-speed-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
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


#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-sp-matrix.h"

using namespace kaldi;


namespace kaldi {

template<typename Real>
std::string NameOf() {
  return (sizeof(Real) == 8 ? "<double>" : "<float>");
}
    
template<typename Real> void TestCuMatrixMatMat(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(dim, dim), N(dim, dim), O(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    O.AddMatMat(1.0, M, kNoTrans, N, kNoTrans, 0.0);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddMatMat" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestSymInvertPosDef(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(dim, dim * 2), N(dim, dim);
  M.SetRandn();
  N.SymAddMat2(1.0, M, kNoTrans, 0.0);
  CuMatrix<Real> Ncopy(N);
  
  int iter = 0;
  Timer tim;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    Ncopy.CopyFromMat(N);
    Ncopy.SymInvertPosDef();
  }
  
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::TestCuInvertPosDef" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixSigmoid(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    N.Sigmoid(M);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Sigmoid" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixSoftmax(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(256, dim), N(256, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    N.ApplySoftMaxPerRow(M);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Softmax" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixTraceMatMat(int32 dim) {
  for (int32 n = 0; n < 2; n++) {
    MatrixTransposeType trans = (n == 0 ? kNoTrans : kTrans);
    BaseFloat time_in_secs = 0.08;
  
    CuMatrix<Real> M(dim, dim), N(dim, dim);
    M.SetRandn();
    N.SetRandn();
    Timer tim;
    int32 iter = 0;
    for (;tim.Elapsed() < time_in_secs; iter++) {
      TraceMatMat(M, N, trans);
    }
    BaseFloat fdim = dim;
    BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
    KALDI_LOG << "For CuMatrix::TraceMatMat" << NameOf<Real>() 
              << (trans == kTrans ? " [transposed]" : "") << ", for dim = "
              << dim << ", speed was " << gflops << " gigaflops.";
  }
}


template<typename Real> void TestCuMatrixCholesky(int32 dim) {
  BaseFloat time_in_secs = 0.08;
  
  CuMatrix<Real> M(dim, dim);
  M.AddToDiag(100.0);  
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++)
    M.Cholesky();

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Cholesky" << NameOf<Real>() 
            << ", for dim = " << dim << ", speed was " << gflops << " gigaflops.";
}



template<typename Real> void TestCuMatrixCopyLowerToUpper(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyLowerToUpper();
  }
  CuMatrix<Real> M2(M, kTrans);
  AssertEqual(M, M2);
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyLowerToUpper" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixCopyFromTp(int32 dim, MatrixTransposeType trans) {
  BaseFloat time_in_secs = 0.025;
  CuTpMatrix<Real> T(dim);
  T.SetRandn();
  CuMatrix<Real> M(dim, dim);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyFromTp(T, trans);
  }
  TpMatrix<Real> T_cpu(T);
  Matrix<Real> M_cpu(T_cpu, trans);
  Matrix<Real> M2_cpu(M);
  AssertEqual(M_cpu, M2_cpu);
  
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyFromTp" << (trans == kNoTrans ? "[NoTrans]":"[Trans]")
            << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixCopyFromSp(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuSpMatrix<Real> S(dim);
  S.SetRandn();
  CuMatrix<Real> M(dim, dim);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyFromSp(S);
  }
  SpMatrix<Real> S_cpu(S);
  Matrix<Real> M_cpu(S_cpu);
  Matrix<Real> M2_cpu(M);
  AssertEqual(M_cpu, M2_cpu);
  
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyFromSp" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}



template<typename Real> void TestCuMatrixCopyUpperToLower(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyUpperToLower();
  }
  CuMatrix<Real> M2(M, kTrans);
  AssertEqual(M, M2);
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyUpperToLower" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixSetZeroAboveDiag(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++)
    M.SetZeroAboveDiag();
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::SetZeroAboveDiag" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void CudaMatrixSpeedTest() {
  std::vector<int32> sizes;
  sizes.push_back(16);
  sizes.push_back(32);
  sizes.push_back(64);
  sizes.push_back(128);
  sizes.push_back(256);
  sizes.push_back(512);
  sizes.push_back(1024);
  int32 ns = sizes.size();
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixMatMat<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestSymInvertPosDef<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCholesky<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSigmoid<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSoftmax<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixTraceMatMat<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyLowerToUpper<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyFromTp<Real>(sizes[s], kNoTrans);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyFromTp<Real>(sizes[s], kTrans);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyFromSp<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyUpperToLower<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSetZeroAboveDiag<Real>(sizes[s]);
}


} // namespace kaldi


int main() {
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif

    kaldi::CudaMatrixSpeedTest<float>();
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CudaMatrixSpeedTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CudaMatrixSpeedTest<double>();
#endif
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  std::cout << "Tests succeeded.\n";
}
