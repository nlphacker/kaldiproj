// cudamatrix/cuda-vector-test.cc

// Copyright 2013 Lucas Ondel
//           2013 Johns Hopkins University (author: Daniel Povey)

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

using namespace kaldi;


namespace kaldi {

/*
 * INITIALIZERS
 */

/*
 * ASSERTS
 */
template<class Real> 
static void AssertEqual(const MatrixBase<Real> &A,
                        const MatrixBase<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++) {
    for (MatrixIndexT j = 0;j < A.NumCols();j++) {
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
  }
}



template<class Real>
static bool ApproxEqual(const MatrixBase<Real> &A,
                        const MatrixBase<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  MatrixBase<Real> diff(A);
  diff.AddSp(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min),
      d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}



template<class Real> 
static void AssertEqual(VectorBase<Real> &A, VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i=0; i < A.Dim(); i++)
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol);
}

template<class Real> 
static void AssertEqual(CuVectorBase<Real> &A, CuVectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i=0; i < A.Dim(); i++)
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol);
}



template<class Real> 
static bool ApproxEqual(VectorBase<Real> &A, VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i=0; i < A.Dim(); i++)
    if (std::abs(A(i)-B(i)) > tol) return false;
  return true;
}

/*
 * Unit tests
 */

template<class Real, class OtherReal> 
static void UnitTestCuVectorCopyFromVec() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    Vector<Real> A(dim);
    A.SetRandn();
    CuVector<OtherReal> B(A);
    Vector<Real> C(B);
    CuVector<Real> D(dim);
    D.CopyFromVec(C);
    Vector<OtherReal> E(dim);
    E.CopyFromVec(D);
    CuVector<Real> F(E);
    CuVector<Real> A2(A);
    KALDI_LOG << "F = " << F;
    KALDI_LOG << "A2 = " << A2;
    KALDI_LOG << "A = " << A;
    KALDI_LOG << "B = " << B;
    KALDI_LOG << "C = " << C;
    KALDI_LOG << "D = " << D;
    KALDI_LOG << "E = " << E;
    AssertEqual(F, A2);
  }
}

template<class Real> 
static void UnitTestCuSubVector() {
  for (int32 iter = 0 ; iter < 10; iter++) {
    int32 M1 = 1 + rand () % 10, M2 = 1 + rand() % 1, M3 = 1 + rand() % 10, M = M1 + M2 + M3,
        m = rand() % M2;
    CuVector<Real> vec(M);
    vec.SetRandn();
    CuSubVector<Real> subvec1(vec, M1, M2),
        subvec2 = vec.Range(M1, M2);
    Real f1 = vec(M1 + m), f2 = subvec1(m), f3 = subvec2(m);
    KALDI_ASSERT(f1 == f2);
    KALDI_ASSERT(f2 == f3);
  }
}



template<class Real> 
static void UnitTestCuVectorMulTp() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    Vector<Real> A(dim);
    A.SetRandn();
    TpMatrix<Real> B(dim);
    B.SetRandn();
    
    CuVector<Real> C(A);
    CuTpMatrix<Real> D(B);

    A.MulTp(B, kNoTrans);
    C.MulTp(D, kNoTrans);

    CuVector<Real> E(A);
    AssertEqual(C, E);
  }
}

template<class Real> 
static void UnitTestCuVectorAddTp() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    Vector<Real> A(dim);
    A.SetRandn();
    TpMatrix<Real> B(dim);
    B.SetRandn();
    Vector<Real> C(dim);
    C.SetRandn();
    
    CuVector<Real> D(A);
    CuTpMatrix<Real> E(B);
    CuVector<Real> F(C); 

    A.AddTpVec(1.0, B, kNoTrans, C, 1.0);
    D.AddTpVec(1.0, E, kNoTrans, F, 1.0);

    CuVector<Real> G(A);
    AssertEqual(D, G);
  }
}

template<class Real> void CuVectorUnitTestVecVec() {
  int32 M = 10 % rand() % 100;
  CuVector<Real> vec1(M), vec2(M);
  vec1.SetRandn();
  vec2.SetRandn();
  Real prod = 0.0;
  for (int32 i = 0; i < M; i++)
    prod += vec1(i) * vec2(i);
  AssertEqual(prod, VecVec(vec1, vec2));
}

template<class Real> void CuVectorUnitTestAddVec() {
  int32 M = 10 % rand() % 100;
  CuVector<Real> vec1(M), vec2(M);
  vec1.SetRandn();
  vec2.SetRandn();
  CuVector<Real> vec1_orig(vec1);
  BaseFloat alpha = 0.43243;
  vec1.AddVec(alpha, vec2);
  
  for (int32 i = 0; i < M; i++)
    AssertEqual(vec1_orig(i) + alpha * vec2(i), vec1(i));
}

template<class Real> void CuVectorUnitTestAddVecExtra() {
  int32 M = 10 % rand() % 100;
  CuVector<Real> vec1(M), vec2(M);
  vec1.SetRandn();
  vec2.SetRandn();
  CuVector<Real> vec1_orig(vec1);
  BaseFloat alpha = 0.43243, beta = 1.4321;
  vec1.AddVec(alpha, vec2, beta);
  
  for (int32 i = 0; i < M; i++)
    AssertEqual(beta * vec1_orig(i) + alpha * vec2(i), vec1(i));
}


template<class Real> void CuVectorUnitTestAddRowSumMat() {
  int32 M = 10 + rand() % 280, N = 10 + rand() % 20;
  BaseFloat alpha = 10.0143432, beta = 43.4321;
  CuMatrix<Real> mat(N, M);
  mat.SetRandn();
  CuVector<Real> vec(M);
  mat.SetRandn();
  Matrix<Real> mat2(mat);
  Vector<Real> vec2(M);
  vec.AddRowSumMat(alpha, mat, beta);
  vec2.AddRowSumMat(alpha, mat2, beta);
  Vector<Real> vec3(vec);
  AssertEqual(vec2, vec3);
}

template<class Real> void CuVectorUnitTestAddColSumMat() {
  int32 M = 10 + rand() % 280, N = 10 + rand() % 20;
  BaseFloat alpha = 10.0143432, beta = 43.4321;
  CuMatrix<Real> mat(M, N);
  mat.SetRandn();
  CuVector<Real> vec(M);
  mat.SetRandn();
  Matrix<Real> mat2(mat);
  Vector<Real> vec2(M);
  vec.AddColSumMat(alpha, mat, beta);
  vec2.AddColSumMat(alpha, mat2, beta);
  Vector<Real> vec3(vec);
  AssertEqual(vec2, vec3);
}


template<class Real> void CuVectorUnitTestApproxEqual() {
  int32 M = 10 + rand() % 100;
  CuVector<Real> vec1(M), vec2(M);
  vec1.SetRandn();
  vec2.SetRandn();
  Real tol = 0.5;
  for (int32 i = 0; i < 10; i++) {
    Real sumsq = 0.0, sumsq_orig = 0.0;
    for (int32 j = 0; j < M; j++) {
      sumsq += (vec1(j) - vec2(j)) * (vec1(j) - vec2(j));
      sumsq_orig += vec1(j) * vec1(j);
    }
    Real rms = sqrt(sumsq), rms_orig = sqrt(sumsq_orig);
    KALDI_ASSERT(vec1.ApproxEqual(vec2, tol) == (rms <= tol * rms_orig));
    tol *= 2.0;
  }
}


template<class Real> void CuVectorUnitTestInvertElements() {
  // Also tests MulElements();
  int32 M = 256 + rand() % 100;
  CuVector<Real> vec1(M);
  vec1.SetRandn();
  CuVector<Real> vec2(vec1);
  vec2.InvertElements();
  CuVector<Real> vec3(vec1);
  vec3.MulElements(vec2);
  // vec3 should be all ones.
  Real prod = VecVec(vec3, vec3);
  AssertEqual(prod, static_cast<Real>(M));
}

template<class Real> void CuVectorUnitTestSum() {
  int32 dim = 256 + 100 % rand();
  CuVector<Real> vec(dim), ones(dim);
  vec.SetRandn();
  ones.Set(1.0);
  KALDI_LOG << "vec is " << vec;
  KALDI_LOG << "ones is " << ones;
  KALDI_LOG << "First 256 is " << VecVec(vec.Range(0, 256), ones.Range(0, 256));
  AssertEqual(VecVec(vec, ones), vec.Sum());
}


template<class Real> void CuVectorUnitTestScale() {
  int32 dim = 10 + 10 % rand();
  CuVector<Real> vec(dim);
  vec.SetRandn();
  CuVector<Real> vec2(vec);
  BaseFloat scale = 0.333;
  vec.Scale(scale);
  KALDI_ASSERT(ApproxEqual(vec(0), vec2(0) * scale));
}

template<class Real> void CuVectorUnitTest() {
  UnitTestCuVectorCopyFromVec<Real, float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported())
#endif
    UnitTestCuVectorCopyFromVec<Real, double>();

  CuVectorUnitTestVecVec<Real>();
  CuVectorUnitTestAddVec<Real>();
  CuVectorUnitTestAddVecExtra<Real>();
  CuVectorUnitTestApproxEqual<Real>();
  CuVectorUnitTestScale<Real>();
  CuVectorUnitTestSum<Real>();
  CuVectorUnitTestInvertElements<Real>();
  CuVectorUnitTestAddRowSumMat<Real>();
  CuVectorUnitTestAddColSumMat<Real>();
  UnitTestCuVectorAddTp<Real>();
  UnitTestCuVectorMulTp<Real>();
  UnitTestCuSubVector<Real>();
}


} // namespace kaldi


int main() {
    //Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(-2); //-2 .. automatic selection
#endif


  kaldi::CuVectorUnitTest<float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported()) {
    kaldi::CuVectorUnitTest<double>();
  } else {
    KALDI_WARN << "Double precision not supported";
  }
#else
  kaldi::CuVectorUnitTest<double>();
#endif
  std::cout << "Tests succeeded.\n";
  return 0;
}