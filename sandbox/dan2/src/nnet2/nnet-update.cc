// nnet2/nnet-update.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-update.h"

namespace kaldi {
namespace nnet2 {


// This class NnetUpdater contains functions for updating the neural net or
// computing its gradient, given a set of NnetTrainingExamples.  Its
// functionality is exported by DoBackprop(), and by ComputeNnetObjf(), so we
// define it in the .cc file.
class NnetUpdater {
 public:
  // Note: in the case of training with SGD, "nnet" and "nnet_to_update" will
  // be identical.  They'll be different if we're accumulating the gradient
  // for a held-out set and don't want to update the model.  Note: nnet_to_update
  // may be NULL if you don't want do do backprop.
  NnetUpdater(const Nnet &nnet,
              Nnet *nnet_to_update);
  
  double ComputeForMinibatch(const std::vector<NnetTrainingExample> &data);
  // returns average objective function over this minibatch.
  
 private:

  /// takes the input and formats as a single matrix, in forward_data_[0].
  void FormatInput(const std::vector<NnetTrainingExample> &data);
  
  // Possibly splices input together from forward_data_[component].
  //   MatrixBase<BaseFloat> &GetSplicedInput(int32 component, Matrix<BaseFloat> *temp_matrix);

  void Propagate();

  /// Computes objective function and derivative at output layer.
  double ComputeObjfAndDeriv(const std::vector<NnetTrainingExample> &data,
                             CuMatrix<BaseFloat> *deriv) const;
  
  /// Returns objf summed (and weighted) over samples.
  /// Note: "deriv" will contain, at input, the derivative w.r.t. the
  /// output layer but will be used as a temporary variable by
  /// this function.
  void Backprop(const std::vector<NnetTrainingExample> &data,
                CuMatrix<BaseFloat> *deriv);

  const Nnet &nnet_;
  Nnet *nnet_to_update_;
  int32 num_chunks_; // same as the minibatch size.
  
  std::vector<CuMatrix<BaseFloat> > forward_data_; // The forward data
  // for the outputs of each of the components.

  // These weights are one per parameter; they equal to the "weight"
  // member variables in the NnetTrainingExample structures.  These
  // will typically be about one on average.
  CuVector<BaseFloat> chunk_weights_;
};

NnetUpdater::NnetUpdater(const Nnet &nnet,
                         Nnet *nnet_to_update):
    nnet_(nnet), nnet_to_update_(nnet_to_update) {
}
 

double NnetUpdater::ComputeForMinibatch(
    const std::vector<NnetTrainingExample> &data) {
  FormatInput(data);
  Propagate();
  CuMatrix<BaseFloat> tmp_deriv;
  double ans = ComputeObjfAndDeriv(data, &tmp_deriv);
  if (nnet_to_update_ != NULL)
    Backprop(data, &tmp_deriv); // this is summed (after weighting), not
                                // averaged.
  return ans;
}

void NnetUpdater::Propagate() {
  static int32 num_times_printed = 0;
        
  int32 num_components = nnet_.NumComponents();
  for (int32 c = 0; c < num_components; c++) {
    const Component &component = nnet_.GetComponent(c);
    const CuMatrix<BaseFloat> &input = forward_data_[c];
    CuMatrix<BaseFloat> &output = forward_data_[c+1];
    // Note: the Propagate function will automatically resize the
    // output.
    component.Propagate(input, num_chunks_, &output);
    // If we won't need the output of the previous layer for
    // backprop, delete it to save memory.
    bool need_last_output =
        (c>0 && nnet_.GetComponent(c-1).BackpropNeedsOutput()) ||
        component.BackpropNeedsInput();
    if (g_kaldi_verbose_level >= 3 && num_times_printed < 100) {
      KALDI_LOG << "Stddev of data for component " << c
                << " for this minibatch is "
                << (TraceMatMat(forward_data_[c], forward_data_[c], kTrans) /
                    (forward_data_[c].NumRows() * forward_data_[c].NumCols()));
      num_times_printed++;
    }
    if (!need_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data.
  }
}

double NnetUpdater::ComputeObjfAndDeriv(
    const std::vector<NnetTrainingExample> &data,
    CuMatrix<BaseFloat> *deriv) const {
  BaseFloat tot_objf = 0.0, tot_weight = 0.0;
  int32 num_components = nnet_.NumComponents();  
  deriv->Resize(num_chunks_, nnet_.OutputDim()); // sets to zero.
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  KALDI_ASSERT(SameDim(output, *deriv));

  // Note: we need to CUDA-ify this block of code somehow, it is
  // becoming a limiting factor.

  std::vector<MatrixElement<BaseFloat> > sv_labels;//(num_chunks_ * data[m].labels.size());
  //int32 cur_index = 0;
  for (int32 m = 0; m < num_chunks_; m++) {
    for (size_t i = 0; i < data[m].labels.size(); i++) {
      MatrixElement<BaseFloat> 
         tmp = {m, data[m].labels[i].first, data[m].labels[i].second};
      sv_labels.push_back(tmp);
      //(sv_labels.Data() +cur_index)->m = m;
      //(sv_labels.Data() +cur_index)->label = data[m].labels[i].first;
      //(sv_labels.Data() +cur_index)->weight = data[m].labels[i].second;
    }
  }

  deriv->CompObjfAndDeriv(sv_labels, output, &tot_objf, &tot_weight);

  KALDI_VLOG(4) << "Objective function is " << (tot_objf/tot_weight) << " over "
                << tot_weight << " samples (weighted).";
  return tot_objf;
}


void NnetUpdater::Backprop(const std::vector<NnetTrainingExample> &data,
                           CuMatrix<BaseFloat> *deriv) {
  int32 num_chunks = data.size();
  // We assume ComputeObjfAndDeriv has already been called.
  for (int32 c = nnet_.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet_.GetComponent(c);
    Component *component_to_update = (nnet_to_update_ == NULL ? NULL :
                                      &(nnet_to_update_->GetComponent(c)));
    CuMatrix<BaseFloat> &input = forward_data_[c],
                     &output = forward_data_[c+1];
    CuMatrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());
    const CuMatrix<BaseFloat> &output_deriv(*deriv);

    component.Backprop(input, output, output_deriv, num_chunks,
                       component_to_update, &input_deriv);
    input_deriv.Swap(deriv);
  }
}


void NnetUpdater::FormatInput(const std::vector<NnetTrainingExample> &data) {
  KALDI_ASSERT(data.size() > 0);
  int32 num_splice = nnet_.LeftContext() + 1 + nnet_.RightContext();
  KALDI_ASSERT(data[0].input_frames.NumRows() >= num_splice);
  
  int32 feat_dim = data[0].input_frames.NumCols(),
         spk_dim = data[0].spk_info.Dim(),
         tot_dim = feat_dim + spk_dim; // we append these at the neural net
                                       // input... note, spk_dim might be 0.
  KALDI_ASSERT(tot_dim == nnet_.InputDim());
  KALDI_ASSERT(data[0].left_context >= nnet_.LeftContext());
  int32 ignore_frames = data[0].left_context - nnet_.LeftContext(); // If
  // the NnetTrainingExample has more left-context than we need, ignore some.
  // this may happen in settings where we increase the amount of context during
  // training, e.g. by adding layers that require more context.
  num_chunks_ = data.size();
  
  forward_data_.resize(nnet_.NumComponents() + 1);

  // First copy to a single matrix on the CPU, so we can copy to
  // GPU with a single copy command.
  Matrix<BaseFloat> temp_forward_data(num_splice * num_chunks_,
                                      tot_dim);
  
  for (int32 chunk = 0; chunk < num_chunks_; chunk++) {
    SubMatrix<BaseFloat> dest(temp_forward_data,
                              chunk * num_splice, num_splice,
                              0, feat_dim);

    Matrix<BaseFloat> full_src(data[chunk].input_frames);
    SubMatrix<BaseFloat> src(full_src, ignore_frames, num_splice, 0, feat_dim);
                             
    dest.CopyFromMat(src);
    if (spk_dim != 0) {
      SubMatrix<BaseFloat> spk_dest(temp_forward_data,
                                    chunk * num_splice, num_splice,
                                    feat_dim, spk_dim);
      spk_dest.CopyRowsFromVec(data[chunk].spk_info);
    }
  }
  forward_data_[0].Swap(&temp_forward_data); // Copy to GPU, if being used.
}

BaseFloat TotalNnetTrainingWeight(const std::vector<NnetTrainingExample> &egs) {
  double ans = 0.0;
  for (size_t i = 0; i < egs.size(); i++)
    for (size_t j = 0; j < egs[i].labels.size(); j++)
      ans += egs[i].labels[j].second;
  return ans;
}

double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetTrainingExample> &examples,
                  Nnet *nnet_to_update) {
  try {
    KALDI_ASSERT(nnet_to_update != NULL && "Call ComputeNnetObjf() instead.");
    NnetUpdater updater(nnet, nnet_to_update);
    return updater.ComputeForMinibatch(examples);
  } catch (...) {
    KALDI_LOG << "Error doing backprop, nnet info is: " << nnet.Info();
    throw;
  }
}

double ComputeNnetObjf(const Nnet &nnet,
                       const std::vector<NnetTrainingExample> &examples) {
  NnetUpdater updater(nnet, NULL);
  return updater.ComputeForMinibatch(examples);
}


double ComputeNnetGradient(
    const Nnet &nnet,
    const std::vector<NnetTrainingExample> &validation_set,
    int32 batch_size,
    Nnet *gradient) {
  bool treat_as_gradient = true;
  gradient->SetZero(treat_as_gradient);
  std::vector<NnetTrainingExample> batch;
  batch.reserve(batch_size);
  double tot_objf = 0.0;
  for (int32 start_pos = 0;
       start_pos < static_cast<int32>(validation_set.size());
       start_pos += batch_size) {
    batch.clear();
    for (int32 i = start_pos;
         i < std::min(start_pos + batch_size,
                      static_cast<int32>(validation_set.size()));
         i++) {
      batch.push_back(validation_set[i]);
    }
    tot_objf += DoBackprop(nnet,
                           batch,
                           gradient);
  }
  return tot_objf / validation_set.size();
}

double ComputeNnetObjf(
    const Nnet &nnet,
    const std::vector<NnetTrainingExample> &validation_set,
    int32 batch_size) {
  std::vector<NnetTrainingExample> batch;
  batch.reserve(batch_size);
  double tot_objf = 0.0;
  for (int32 start_pos = 0;
       start_pos < static_cast<int32>(validation_set.size());
       start_pos += batch_size) {
    batch.clear();
    for (int32 i = start_pos;
         i < std::min(start_pos + batch_size,
                      static_cast<int32>(validation_set.size()));
         i++) {
      batch.push_back(validation_set[i]);
    }
    tot_objf += ComputeNnetObjf(nnet, batch);
  }
  return tot_objf;
}

  
  
} // namespace nnet2
} // namespace kaldi
