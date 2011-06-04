// transform/regtree-mllr-diag-gmm-test.cc

// Copyright 2009-2011   Arnab Ghoshal (Saarland University)

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

#include "util/common-utils.h"
#include "gmm/diag-gmm.h"
#include "gmm/estimate-diag-gmm.h"
#include "gmm/estimate-am-diag-gmm.h"
#include "gmm/model-test-common.h"
#include "transform/regtree-mllr-diag-gmm.h"

using kaldi::int32;
using kaldi::BaseFloat;
using kaldi::RegtreeMllrDiagGmmAccs;
namespace ut = kaldi::unittest;

void TestMllrAccsIO(const kaldi::AmDiagGmm &am_gmm,
                    const kaldi::RegressionTree &regtree,
                    const RegtreeMllrDiagGmmAccs &accs,
                    const kaldi::Matrix<BaseFloat> adapt_data) {
  // First, non-binary write
  KALDI_LOG << "Test ASCII IO.";
  accs.Write(kaldi::Output("tmpf", false).Stream(), false);

  kaldi::RegtreeMllrDiagGmm mllr;
  kaldi::RegtreeMllrOptions opts;
  opts.min_count = 100;
  opts.use_regtree =false;
  accs.Update(regtree, opts, &mllr, NULL, NULL);
  kaldi::AmDiagGmm am0;
  am0.CopyFromAmDiagGmm(am_gmm);
  mllr.TransformModel(regtree, &am0);

  BaseFloat loglike = 0;
  int32 npoints = adapt_data.NumRows();
  for (int32 j = 0; j < npoints; j++) {
    loglike += am0.LogLikelihood(0, adapt_data.Row(j));
  }
  KALDI_LOG << "Per-frame loglike after adaptation = " << (loglike/npoints)
            << " over " << npoints << " frames.";

  size_t num_comp2 = 1 + kaldi::RandInt(0, 9);  // random number of mixtures
  int32 dim = am_gmm.Dim();
  kaldi::DiagGmm gmm2;
  ut::InitRandDiagGmm(dim, num_comp2, &gmm2);
  kaldi::Vector<BaseFloat> data(dim);
  gmm2.Generate(&data);
  BaseFloat loglike0 = am0.LogLikelihood(0, data);

  bool binary_in;
  kaldi::RegtreeMllrDiagGmm mllr1;
  RegtreeMllrDiagGmmAccs *accs1 = new RegtreeMllrDiagGmmAccs();
  // Non-binary read
  kaldi::Input ki1("tmpf", &binary_in);
  accs1->Read(ki1.Stream(), binary_in, false);
  accs1->Update(regtree, opts, &mllr1, NULL, NULL);
  delete accs1;
  kaldi::AmDiagGmm am1;
  am1.CopyFromAmDiagGmm(am_gmm);
  mllr.TransformModel(regtree, &am1);
  BaseFloat loglike1 = am1.LogLikelihood(0, data);
  KALDI_LOG << "LL0 = " << loglike0 << "; LL1 = " << loglike1;
  kaldi::AssertEqual(loglike0, loglike1, 1e-6);

  kaldi::RegtreeMllrDiagGmm mllr2;
  // Next, binary write
  KALDI_LOG << "Test Binary IO.";
  accs.Write(kaldi::Output("tmpfb", true).Stream(), true);
  RegtreeMllrDiagGmmAccs *accs2 = new RegtreeMllrDiagGmmAccs();
  // Binary read
  kaldi::Input ki2("tmpfb", &binary_in);
  accs2->Read(ki2.Stream(), binary_in, false);
  accs2->Update(regtree, opts, &mllr2, NULL, NULL);
  delete accs2;
  kaldi::AmDiagGmm am2;
  am2.CopyFromAmDiagGmm(am_gmm);
  mllr.TransformModel(regtree, &am2);
  BaseFloat loglike2 = am2.LogLikelihood(0, data);
  KALDI_LOG << "LL0 = " << loglike0 << "; LL2 = " << loglike2;
  kaldi::AssertEqual(loglike0, loglike2, 1e-6);
}

void UnitTestRegtreeMllrDiagGmm() {
  size_t dim = 1 + kaldi::RandInt(1, 9);  // random dimension of the gmm
  size_t num_comp = 1 + kaldi::RandInt(0, 9);  // random number of mixtures
  kaldi::DiagGmm gmm;
  ut::InitRandDiagGmm(dim, num_comp, &gmm);
  kaldi::AmDiagGmm am_gmm;
  am_gmm.Init(gmm, 1);

  size_t num_comp2 = 1 + kaldi::RandInt(0, 9);  // random number of mixtures
  kaldi::DiagGmm gmm2;
  ut::InitRandDiagGmm(dim, num_comp2, &gmm2);
  int32 npoints = dim*(dim+1)*2 + rand() % 100 + 500;
  kaldi::Matrix<BaseFloat> adapt_data(npoints, dim);
  for (int32 j = 0; j < npoints; j++) {
    kaldi::SubVector<BaseFloat> row(adapt_data, j);
    gmm2.Generate(&row);
  }

  kaldi::RegressionTree regtree;
  std::vector<int32> sil_indices;
  kaldi::Vector<BaseFloat> state_occs(1);
  state_occs(0) = npoints;
  regtree.BuildTree(state_occs, sil_indices, am_gmm, 24);
  int32 num_bclass = regtree.NumBaseclasses();

  kaldi::RegtreeMllrDiagGmmAccs accs;
  BaseFloat loglike = 0;
  accs.Init(num_bclass, dim);
  for (int32 j = 0; j < npoints; j++) {
    loglike += accs.AccumulateForGmm(regtree, am_gmm, adapt_data.Row(j),
                                     0, 1.0);
  }
  KALDI_LOG << "Per-frame loglike during accumulations = " << (loglike/npoints)
            << " over " << npoints << " frames.";

  TestMllrAccsIO(am_gmm, regtree, accs, adapt_data);
}

int main() {
  for (int i = 0; i <= 10; i++)
    UnitTestRegtreeMllrDiagGmm();
  std::cout << "Test OK.\n";
}

