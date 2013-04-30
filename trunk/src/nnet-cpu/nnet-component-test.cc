// nnet-cpu/net-component-test.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet-cpu/nnet-component.h"
#include "util/common-utils.h"

namespace kaldi {


void UnitTestGenericComponentInternal(const Component &component) {
  int32 input_dim = component.InputDim(),
       output_dim = component.OutputDim();


  KALDI_LOG << component.Info();

  Vector<BaseFloat> objf_vec(output_dim); // objective function is linear function of output.
  objf_vec.SetRandn(); // set to Gaussian noise.
  
  int32 num_egs = 10 + rand() % 5;
  Matrix<BaseFloat> input(num_egs, input_dim),
      output(num_egs, output_dim);
  input.SetRandn();
  
  component.Propagate(input, 1, &output);
  {
    bool binary = (rand() % 2 == 0);
    Output ko("tmpf", binary);
    component.Write(ko.Stream(), binary);
  }
  Component *component_copy;
  {
    bool binary_in;
    Input ki("tmpf", &binary_in);
    component_copy = Component::ReadNew(ki.Stream(), binary_in);
  }
  
  { // Test backward derivative is correct.
    Vector<BaseFloat> output_objfs(num_egs);
    output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec, 0.0);
    BaseFloat objf = output_objfs.Sum();

    Matrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
    for (int32 i = 0; i < output_deriv.NumRows(); i++)
      output_deriv.Row(i).CopyFromVec(objf_vec);

    Matrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());

    Matrix<BaseFloat> empty_mat;
    Matrix<BaseFloat> &input_ref =
        (component_copy->BackpropNeedsInput() ? input : empty_mat),
        &output_ref =
        (component_copy->BackpropNeedsOutput() ? output : empty_mat);
    int32 num_chunks = 1;
    component_copy->Backprop(input_ref, output_ref,
                             output_deriv, num_chunks, NULL, &input_deriv);

    int32 num_ok = 0, num_bad = 0, num_tries = 7;
    KALDI_LOG << "Comparing feature gradients " << num_tries << " times.";
    for (int32 i = 0; i < num_tries; i++) {
      Matrix<BaseFloat> perturbed_input(input.NumRows(), input.NumCols());
      perturbed_input.SetRandn();
      perturbed_input.Scale(1.0e-04); // scale by a small amount so it's like a delta.
      BaseFloat predicted_difference = TraceMatMat(perturbed_input,
                                                   input_deriv, kTrans);
      perturbed_input.AddMat(1.0, input); // now it's the input + a delta.
      { // Compute objf with perturbed input and make sure it matches
        // prediction.
        Matrix<BaseFloat> perturbed_output(output.NumRows(), output.NumCols());
        component.Propagate(perturbed_input, 1, &perturbed_output);
        Vector<BaseFloat> perturbed_output_objfs(num_egs);
        perturbed_output_objfs.AddMatVec(1.0, perturbed_output, kNoTrans,
                                         objf_vec, 0.0);
        BaseFloat perturbed_objf = perturbed_output_objfs.Sum(),
             observed_difference = perturbed_objf - objf;
        KALDI_LOG << "Input gradients: comparing " << predicted_difference
                  << " and " << observed_difference;
        if (fabs(predicted_difference - observed_difference) >
            0.1 * fabs((predicted_difference + observed_difference)/2)) {
          KALDI_WARN << "Bad difference!";
          num_bad++;
        } else {
          num_ok++;
        }
      }
    }
    KALDI_LOG << "Succeeded for " << num_ok << " out of " << num_tries
              << " tries.";
    KALDI_ASSERT(num_ok > num_bad);
  }

  UpdatableComponent *ucomponent =
      dynamic_cast<UpdatableComponent*>(component_copy);

  if (ucomponent != NULL) { // Test parameter derivative is correct.

    int32 num_ok = 0, num_bad = 0, num_tries = 10;
    KALDI_LOG << "Comparing model gradients " << num_tries << " times.";
    for (int32 i = 0; i < num_tries; i++) {    
      UpdatableComponent *perturbed_ucomponent =
          dynamic_cast<UpdatableComponent*>(ucomponent->Copy()),
          *gradient_ucomponent =
          dynamic_cast<UpdatableComponent*>(ucomponent->Copy());
      KALDI_ASSERT(perturbed_ucomponent != NULL);
      gradient_ucomponent->SetZero(true); // set params to zero and treat as gradient.
      BaseFloat perturb_stddev = 1.0e-04;
      perturbed_ucomponent->PerturbParams(perturb_stddev);

      Vector<BaseFloat> output_objfs(num_egs);
      output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec, 0.0);
      BaseFloat objf = output_objfs.Sum();

      Matrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
      for (int32 i = 0; i < output_deriv.NumRows(); i++)
        output_deriv.Row(i).CopyFromVec(objf_vec);
      Matrix<BaseFloat> input_deriv; // (input.NumRows(), input.NumCols());

      int32 num_chunks = 1;

      // This will compute the parameter gradient.
      ucomponent->Backprop(input, output, output_deriv, num_chunks,
                           gradient_ucomponent, &input_deriv);

      // Now compute the perturbed objf.
      BaseFloat objf_perturbed;
      {
        Matrix<BaseFloat> output_perturbed; // (num_egs, output_dim);
        perturbed_ucomponent->Propagate(input, 1, &output_perturbed);
        Vector<BaseFloat> output_objfs_perturbed(num_egs);
        output_objfs_perturbed.AddMatVec(1.0, output_perturbed,
                                         kNoTrans, objf_vec, 0.0);
        objf_perturbed = output_objfs_perturbed.Sum();
      }

      BaseFloat delta_objf_observed = objf_perturbed - objf,
          delta_objf_predicted = (perturbed_ucomponent->DotProduct(*gradient_ucomponent) -
                                  ucomponent->DotProduct(*gradient_ucomponent));
      
      KALDI_LOG << "Model gradients: comparing " << delta_objf_observed
                << " and " << delta_objf_predicted;
      if (fabs(delta_objf_predicted - delta_objf_observed) >
          0.05 * fabs((delta_objf_predicted + delta_objf_observed)/2)) {
        KALDI_WARN << "Bad difference!";
        num_bad++;
      } else {
        num_ok++;
      }
      delete perturbed_ucomponent;
      delete gradient_ucomponent;
    }
    KALDI_ASSERT(num_ok > num_bad);
  }
  delete component_copy; // No longer needed.
}


void UnitTestSigmoidComponent() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.
  
  int32 input_dim = 10 + rand() % 50;
  {
    SigmoidComponent sigmoid_component(input_dim);
    UnitTestGenericComponentInternal(sigmoid_component);
  }
  {
    SigmoidComponent sigmoid_component;
    sigmoid_component.InitFromString("dim=15");
    UnitTestGenericComponentInternal(sigmoid_component);
  }
}

void UnitTestReduceComponent() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.
  
  int32 input_dim = 10 + rand() % 50, n = 1 + rand() % 3;
  {
    ReduceComponent reduce_component(input_dim, n);
    UnitTestGenericComponentInternal(reduce_component);
  }
  {
    ReduceComponent reduce_component;
    reduce_component.InitFromString("dim=15 n=3");
    UnitTestGenericComponentInternal(reduce_component);
  }
}


template<class T>
void UnitTestGenericComponent() { // works if it has an initializer from int,
  // e.g. tanh, sigmoid.
  
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.
  
  int32 input_dim = 10 + rand() % 50;
  {
    T component(input_dim);
    UnitTestGenericComponentInternal(component);
  }
  {
    T component;
    component.InitFromString("dim=15");
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestAffineComponent() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0;
  int32 input_dim = 5 + rand() % 10, output_dim = 5 + rand() % 10;
  bool precondition = (rand() % 2 == 1);
  {
    AffineComponent component;
    component.Init(learning_rate, input_dim, output_dim,
                   param_stddev, bias_stddev, precondition);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=10 output-dim=15 param-stddev=0.1";
    AffineComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestAffineComponentPreconditioned() {
  BaseFloat learning_rate = 0.01,
             param_stddev = 0.1, bias_stddev = 1.0, alpha = 0.01;
  int32 input_dim = 5 + rand() % 10, output_dim = 5 + rand() % 10;
  bool precondition = (rand() % 2 == 1);
  {
    AffineComponentPreconditioned component;
    component.Init(learning_rate, input_dim, output_dim,
                   param_stddev, bias_stddev, precondition,
                   alpha);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=16 output-dim=15 param-stddev=0.1 alpha=0.01";
    AffineComponentPreconditioned component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestAffinePreconInputComponent() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0,
      avg_samples = 100.0;
  int32 input_dim = 5 + rand() % 10, output_dim = 5 + rand() % 10;

  {
    AffinePreconInputComponent component;
    component.Init(learning_rate, input_dim, output_dim,
                   param_stddev, bias_stddev, avg_samples);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=10 output-dim=15 param-stddev=0.1 avg-samples=100";
    AffinePreconInputComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestBlockAffineComponent() {
  BaseFloat learning_rate = 0.01,
      param_stddev = 0.1, bias_stddev = 1.0;
  int32 num_blocks = 1 + rand() % 3,
         input_dim = num_blocks * (2 + rand() % 4),
        output_dim = num_blocks * (2 + rand() % 4);
  
  {
    BlockAffineComponent component;
    component.Init(learning_rate, input_dim, output_dim,
                   param_stddev, bias_stddev, num_blocks);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 input-dim=10 output-dim=15 param-stddev=0.1 num-blocks=5";
    BlockAffineComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestMixtureProbComponent() {
  BaseFloat learning_rate = 0.01,
      diag_element = 0.8;
  std::vector<int32> sizes;
  int32 num_sizes = 1 + rand() % 5; // allow 
  for (int32 i = 0; i < num_sizes; i++)
    sizes.push_back(1 + rand() % 5);
  
  
  {
    MixtureProbComponent component;
    component.Init(learning_rate, diag_element, sizes);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "learning-rate=0.01 diag-element=0.9 dims=3:4:5";
    MixtureProbComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestDctComponent() {
  int32 m = 1 + rand() % 4, n = 1 + rand() % 4,
  dct_dim = m, dim = m * n;
  bool reorder = (rand() % 2 == 0);
  {
    DctComponent component;
    component.Init(dim, dct_dim, reorder);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true dct-keep-dim=1";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true dct-keep-dim=2";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true dct-keep-dim=3";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "dim=10 dct-dim=5 reorder=true dct-keep-dim=4";
    DctComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}


void UnitTestFixedLinearComponent() {
  int32 m = 1 + rand() % 4, n = 1 + rand() % 4;
  {
    Matrix<BaseFloat> mat(m, n);
    FixedLinearComponent component;
    component.Init(mat);
    UnitTestGenericComponentInternal(component);
  }
}



void UnitTestParsing() {
  int32 i;
  BaseFloat f;
  bool b;
  std::vector<int32> v;
  std::string s = "x=y";
  KALDI_ASSERT(ParseFromString("foo", &s, &i) == false
               && s == "x=y");
  KALDI_ASSERT(ParseFromString("foo", &s, &f) == false
               && s == "x=y");
  KALDI_ASSERT(ParseFromString("foo", &s, &v) == false
               && s == "x=y");
  KALDI_ASSERT(ParseFromString("foo", &s, &b) == false
               && s == "x=y");
  {
    std::string s = "x=1";
    KALDI_ASSERT(ParseFromString("x", &s, &i) == true
                 && i == 1 && s == "");
    s = "a=b x=1";
    KALDI_ASSERT(ParseFromString("x", &s, &i) == true
                 && i == 1 && s == "a=b");
  }
  {
    std::string s = "foo=false";
    KALDI_ASSERT(ParseFromString("foo", &s, &b) == true
                 && b == false && s == "");
    s = "x=y foo=true a=b";
    KALDI_ASSERT(ParseFromString("foo", &s, &b) == true
                 && b == true && s == "x=y a=b");    
  }

  {
    std::string s = "foobar x=1";
    KALDI_ASSERT(ParseFromString("x", &s, &f) == true
                 && f == 1.0 && s == "foobar");
    s = "a=b x=1 bxy";
    KALDI_ASSERT(ParseFromString("x", &s, &f) == true
                 && f == 1.0 && s == "a=b bxy");
  }
  {
    std::string s = "x=1:2:3";
    KALDI_ASSERT(ParseFromString("x", &s, &v) == true
                 && v.size() == 3 && v[0] == 1 && v[1] == 2 && v[2] == 3
                 && s == "");
    s = "a=b x=1:2:3 c=d";
    KALDI_ASSERT(ParseFromString("x", &s, &v) == true
                 && f == 1.0 && s == "a=b c=d");
  }

}

int BasicDebugTestForSplice (bool output=false) {
  int32 C=5;
  int32 K=4, contextLen=1;
  int32 R=3+2 * contextLen;
 
  SpliceComponent *c = new SpliceComponent();
  c->Init(C, contextLen, contextLen, K);
  Matrix<BaseFloat> in(R, C);
  Matrix<BaseFloat> out(R, c->OutputDim());

  in.SetRandn();
  if (output)
    KALDI_WARN << in;

  c->Propagate(in, 1, &out);
  
  if (output) 
    KALDI_WARN << out;

  out.Set(1);
  
  if (K > 0) {
    SubMatrix<BaseFloat> k(out, 0, out.NumRows(), c->OutputDim() - K, K);
    k.Set(-2);
  }

  if (output)
    KALDI_WARN << out;
  
  int32 num_chunks = 1;
  c->Backprop(in, in, out, num_chunks, c, &in);
  
  if (output)
    KALDI_WARN << in ;

  return 0;
}

} // namespace kaldi

#include "matrix/matrix-functions.h"


int main() {
  using namespace kaldi;

  BasicDebugTestForSplice(true);

  for (int32 i = 0; i < 5; i++) {
    UnitTestGenericComponent<SigmoidComponent>();
    UnitTestGenericComponent<TanhComponent>();
    UnitTestGenericComponent<PermuteComponent>();
    UnitTestGenericComponent<SoftmaxComponent>();
    UnitTestGenericComponent<RectifiedLinearComponent>();
    UnitTestGenericComponent<SoftHingeComponent>();
    UnitTestSigmoidComponent();
    UnitTestReduceComponent();
    UnitTestAffineComponent();
    UnitTestAffinePreconInputComponent();
    UnitTestBlockAffineComponent();
    UnitTestMixtureProbComponent();
    UnitTestDctComponent();
    UnitTestFixedLinearComponent();
    UnitTestAffineComponentPreconditioned();
    UnitTestParsing();
  }
}
