#include <iostream>
#include "base/kaldi-common.h"


#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-packed-matrix.h"
#include <numeric>
#include <time.h>

namespace kaldi {

/*
 * INITIALIZERS
 */ 
template<class Real>
static void InitRand(SpMatrix<Real> *M) {
  do {
    for (MatrixIndexT i = 0; i < M->NumRows(); i++) {
      for (MatrixIndexT j = 0; j <= i; j++ ) {
	(*M)(i,j) = RandGauss();
      }
    }
  } while (M->NumRows() != 0 && M->Cond() > 100);
}

/*
 * ASSERTS
 */
template<class Real> 
static void ApproxEqual(const SpMatrix<Real> &A,
			const SpMatrix<Real> &B,
			float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
}

// ApproxEqual
template<class Real>
static bool ApproxEqual(const SpMatrix<Real> &A,
                        const SpMatrix<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  SpMatrix<Real> diff(A);
  diff.AddSp(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min),
    d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}

template<class Real>
static void SimpleTest() {
  // test differnet constructors + CopyFrom* + CopyToMat
  int32 dim = 5 + rand() % 10;
  std::cout << "dim is : " << dim << std::endl;
  CuPackedMatrix<Real> A(dim);
  PackedMatrix<Real> B(dim);
  A.CopyToMat(&B);
  std::cout << "The cudamatrix A is" << std::endl;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << B(i,j) << " ";
    }
    std::cout << std::endl;
  }
  for (int i = 0; i < dim; i++) {
    B(i,i) = i;
  }
  CuPackedMatrix<Real> C(B);
  PackedMatrix<Real> D(dim);
  C.CopyToMat(&D);
  std::cout << "The cudamatrix C is" << std::endl;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << D(i,j) << " ";
      D(i,j) = D(i,j) + 1;
    }
    std::cout << std::endl;
  }
  C.CopyFromPacked(D);
  CuPackedMatrix<Real> E(C);
  PackedMatrix<Real> F(dim);
  E.CopyToMat(&F);
  std::cout << "The cudamatrix E is" << std::endl;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << F(i,j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Trace(E) = " << E.Trace() << std::endl;
  /*
  CuSpMatrix<Real> A(dim);
  SpMatrix<Real> B(dim);
  A.CopyToMat(&B);
  B(1,1)=14;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << B(i,j) << " ";
    }
    std::cout << std::endl;
  }
  SpMatrix<Real> C(dim);
  InitRand(&C);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << C(i,j) << " ";
    }
    std::cout << std::endl;
  }
  C(7,7) = 22;
  CuSpMatrix<Real> D(C);
  SpMatrix<Real> E(dim);
  D.CopyToMat(&E);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << E(i,j) << " ";
    }
    std::cout << std::endl;
  }
  E(0,0) = 13;
  D.CopyFromPacked(E);
  CuSpMatrix<Real> F(D);
  SpMatrix<Real> G(dim);
  F.CopyToMat(&G);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << G(i,j) << " ";
    }
    std::cout << std::endl;
  }
  */
}

template<class Real> static void UnitTestCopySp() {
  // Checking that the various versions of copying                                 
  // matrix to SpMatrix work the same in the symmetric case.                         
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    int32 dim = 5 + rand() %  10;
    CuSpMatrix<Real> S(dim), T(dim);
    S.SetRandn();
    Matrix<Real> M(S);
    T.CopyFromMat(M, kTakeMeanAndCheck);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeMean);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeLower);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeUpper);
    AssertEqual(S, T);
  }
}

template<class Real>
static void CuMatrixUnitTest(bool full_test) {
  SimpleTest<Real>();
}
} //namespace

int main() {
  using namespace kaldi;
  bool full_test = false;
  kaldi::CuMatrixUnitTest<double>(full_test);
  KALDI_LOG << "Tests succeeded.\n";
  return 0;
}