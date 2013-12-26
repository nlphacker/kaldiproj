// feat/pitch-functions.cc

// Copyright    2013  Pegah Ghahremani

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
#include <limits>
#include "feat/feature-functions.h"
#include "matrix/matrix-functions.h"
#include "feat/pitch-functions.h"
#include "feat/mel-computations.h"

namespace kaldi {

// compute the Weighted Moving Window normalization
// Subtract the weighted-moving average over a largish window
// The weight is equal to probability of voicing per frame.
void WeightedMwn(int32 normalization_window_size,
                 const MatrixBase<BaseFloat> &input,
                 Matrix<BaseFloat> *output) {
  int32 num_frames = input.NumRows(), dim = input.NumCols();
  KALDI_ASSERT(dim == 2);
  int32 last_window_start = -1, last_window_end = -1;
  Vector<BaseFloat> cur_sum(dim), cur_sumsq(dim);
  Vector<BaseFloat> weighted_sum(1);
  BaseFloat pov_sum = 0.0;
  for (int32 t = 0; t < num_frames; t++) {
    int32 window_start, window_end;
    window_start = t - (normalization_window_size/2);
    window_end = window_start + normalization_window_size;

    if (window_start < 0) {
      window_end -=window_start;
      window_start = 0;
    }

    if (window_end > num_frames) {
      window_start -= (window_end - num_frames);
      window_end = num_frames;
      if (window_start < 0) window_start = 0;
    }
    if (last_window_start == -1) {
      SubMatrix<BaseFloat> pitch_part(input, window_start,
                                      window_end - window_start,
                                      1, 1);
      SubMatrix<BaseFloat> pov_part(input, window_start,
                                    window_end - window_start,
                                    0, 1);
      Matrix<BaseFloat> pov_pitch(1, 1);
      pov_pitch.AddMatMat(1.0, pitch_part, kTrans, pov_part, kNoTrans, 0.0);

      // sum of pov
      pov_sum =  pov_part.Sum();

      // weighted sum of pitch
      weighted_sum.AddRowSumMat(1.0, pov_pitch, 0.0);
      cur_sum(1) = weighted_sum(0);
    } else {
      if (window_start > last_window_start) {
        KALDI_ASSERT(window_start == last_window_start + 1);
        SubVector<BaseFloat> frame_to_remove(input, last_window_start);
        pov_sum -= frame_to_remove(0);
        weighted_sum(0) -= frame_to_remove(0) * frame_to_remove(1);
      }
      if (window_end > last_window_end) {
        KALDI_ASSERT(window_end == last_window_end + 1);
        SubVector<BaseFloat> frame_to_add(input, last_window_end);
        pov_sum += frame_to_add(0);
        weighted_sum(0) += frame_to_add(0) * frame_to_add(1);
      }
      cur_sum(1) = weighted_sum(0);
    }

    int32 window_frames = window_end - window_start;
    last_window_start = window_start;
    last_window_end = window_end;
    KALDI_ASSERT(window_frames > 0);
    SubVector<BaseFloat> input_frame(input, t),
        output_frame(*output, t);
    output_frame.CopyFromVec(input_frame);
    output_frame.AddVec(-1.0/pov_sum, cur_sum);
    output_frame(0) = input_frame(0);
    KALDI_ASSERT((*output)(t, 1) - (*output)(t, 1) == 0)
        }
}
// it would process the raw pov using some nonlinearity
// if apply_sigmoid, it would map the pov to [0, 1] using sigmoid function
// nonlin 1 : power function as nonlineariy
//        2 : new nonlinearty for pov (coeffs trained by keele)
// apply_sigmoid to map the pov to [0, 1] using sigmoid function
void ProcessPovFeatures(Matrix<BaseFloat> *input,
                        int nonlin,
                        bool apply_sigmoid) {
  if (nonlin != 1 && nonlin != 2) KALDI_ERR << " nonlin should be 1 or 2";
  int32 num_frames = input->NumRows();
  if (nonlin == 1) {
    for (int32 i = 0; i < num_frames; i++) {
      BaseFloat p = (*input)(i, 0);
      if (p > 1.0) {
        p = 1.0;
      } else if (p < -1.0) {
        p = -1.0;
      }
      (*input)(i, 0) = pow((1.0001 - p), 0.15) - 1.0;
      KALDI_ASSERT((*input)(i, 0) - (*input)(i, 0) == 0);
    }
  } else if (nonlin == 2) {
    for (int32 i = 0; i < num_frames; i++) {
      BaseFloat p = fabs((*input)(i, 0));
      if (p > 1.0)
        p = 1.0;
      p = -5.2 + 5.4 * exp(7.5 * (p - 1.0)) +
        4.8 * p -2.0 * exp(-10.0 * p)+4.2 * exp(20.0 * (p - 1.0));
      if (apply_sigmoid)
        p = 1.0/(1 + exp(-1.0 * p));
      (*input)(i, 0) = p;
      KALDI_ASSERT((*input)(i, 0) - (*input)(i, 0) == 0);
    }
  }
}

void TakeLogOfPitch(Matrix<BaseFloat> *input) {
  int32 num_frames = input->NumRows();
  for (int32 i = 0; i < num_frames; i++) {
    BaseFloat p = (*input)(i, 1);
    KALDI_ASSERT((*input)(i, 1) > 0.0);
    (*input)(i, 1) = log(p);
  }
}

int32 PitchNumFrames(int32 nsamp,
                     const PitchExtractionOptions &opts) {
  int32 frame_shift = opts.NccfWindowShift();
  int32 frame_length = opts.NccfWindowSize();
  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);
  if (static_cast<int32>(nsamp) < frame_length)
    return 0;
  else
    return (1 + ((nsamp - frame_length) / frame_shift));
}
void PreemphasizeFrame(VectorBase<double> *waveform, double preemph_coeff) {
  if (preemph_coeff == 0.0) return;
  assert(preemph_coeff >= 0.0 && preemph_coeff <= 1.0);
  for (int32 i = waveform->Dim()-1; i > 0; i--)
    (*waveform)(i) -= preemph_coeff * (*waveform)(i-1);
  (*waveform)(0) -= preemph_coeff * (*waveform)(0);
}

void ExtractFrame(const VectorBase<double> &wave,
                  int32 frame_num,
                  const PitchExtractionOptions &opts,
                  Vector<double> *window) {
  int32 frame_shift = opts.NccfWindowShift();
  int32 frame_length = opts.NccfWindowSize();
  int32 outer_max_lag = round(opts.resample_freq / opts.min_f0) +
      round(opts.lowpass_filter_width/2);

  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);
  int32 start = frame_shift * frame_num;
  int32 end = start + outer_max_lag  + frame_length;
  int32 frame_length_new = frame_length + outer_max_lag;

  if (window->Dim() != frame_length_new)
    window->Resize(frame_length_new);

  SubVector<double> wave_part(wave, start,
                              std::min(frame_length_new, wave.Dim()-start));

  SubVector<double>  window_part(*window, 0,
                                 std::min(frame_length_new, wave.Dim()-start));
  window_part.CopyFromVec(wave_part);
  
  if (opts.preemph_coeff != 0.0)
    PreemphasizeFrame(&window_part, opts.preemph_coeff);

  if (end > wave.Dim())
    SubVector<double>(*window, frame_length_new-end+wave.Dim(),
                      end-wave.Dim()).SetZero();
}

class ArbitraryResample {
 public:
  ArbitraryResample(int32 num_samples_in, double samp_rate_in,
                    double filter_cutoff,
                    const std::vector<double> &sample_points,
                    int32 num_zeros_upsample):
      num_samples_in_(num_samples_in),
      samp_rate_in_(samp_rate_in),
      filter_cutoff_(filter_cutoff),
      num_zeros_upsample_(num_zeros_upsample) {
    KALDI_ASSERT(num_samples_in > 0 && samp_rate_in > 0.0 &&
                 filter_cutoff > 0.0 &&
                 filter_cutoff * 2.0 <= samp_rate_in
                 && num_zeros_upsample > 0);
    // set up weights_ and indices_.  Please try to keep all functions short and
    SetIndex(sample_points);
    SetWeights(sample_points);
  }

  int32 NumSamplesIn() const { return num_samples_in_; }
  int32 NumSamplesOut() const { return indexes_.size(); }

  void Upsample(const MatrixBase<double> &input,
                MatrixBase<double> *output) {
    // each row of "input" corresponds to the data to resample;
    // the corresponding row of "output" is the resampled data.

    KALDI_ASSERT(input.NumRows() == output->NumRows() &&
                 input.NumCols() == num_samples_in_ &&
                 output->NumCols() == indexes_.size());

    Vector<double> output_col(output->NumRows());
    for (int32 i = 0; i < NumSamplesOut(); i++) {
      SubMatrix<double> input_part(input, 0, input.NumRows(),
                                   indexes_[i].first_index, indexes_[i].num_indices);
      const Vector<double> &weight_vec(weights_[i]);
      output_col.AddMatVec(1.0/samp_rate_in_, input_part, kNoTrans, weight_vec, 0.0);
      output->CopyColFromVec(output_col, i);
    }
  }
 private:
  void SetIndex(const std::vector<double> &sample_points) {
    int32 last_ind, num_sample = sample_points.size();
    indexes_.resize(num_sample);
    for (int32  i = 0; i < num_sample; i++) {
      indexes_[i].first_index = std::max(0,
        static_cast<int>(ceil(samp_rate_in_ * (sample_points[i]
        - num_zeros_upsample_/(2.0 * filter_cutoff_)))));
      last_ind = std::min((num_samples_in_ - 1),
        static_cast<int>(floor(samp_rate_in_ *
        (sample_points[i] + num_zeros_upsample_ / (2.0 * filter_cutoff_))) + 1));
      indexes_[i].num_indices = last_ind - indexes_[i].first_index + 1;
    }
  }

  void SetWeights(const std::vector<double> &sample_points) {
    int32 num_samples_out = NumSamplesOut();
    weights_.resize(num_samples_out);
    double t, j_double, tn;
    for (int32 i = 0; i < num_samples_out; i++) {
      weights_[i].Resize(indexes_[i].num_indices);
      for (int32 j = 0 ; j < indexes_[i].num_indices; j++) {
        j_double = static_cast<double>(j);
        t = sample_points[i] - (j_double + indexes_[i].first_index) / samp_rate_in_;
        tn = FilterFunc(t);
        weights_[i](j) = tn;
      }
    }
  }
  double FilterFunc(const double &t) {
    double f_t = 0,  win = 0;

    if (fabs(t) < num_zeros_upsample_ /(2.0 * filter_cutoff_))
      win = 0.5 * (1 + cos(M_2PI * filter_cutoff_ / num_zeros_upsample_ * t));
    if (t != 0)
      f_t = sin(M_2PI * filter_cutoff_ * t) / (M_PI * t) * win;
    else
      f_t = 2 * filter_cutoff_ * win;
    return f_t;
  }
  int32 num_samples_in_;
  double samp_rate_in_;
  double filter_cutoff_;
  int32 num_zeros_upsample_;
  struct IndexInfo {
    int32 first_index;  // The first input-sample index that we sum
                        // over, for this output-sample index.
    int32 num_indices;  // The number of indices that we sum over.
  };
  std::vector<IndexInfo> indexes_;  // indexes_.size() equals sample_points.size().
  std::vector<Vector<double> > weights_;  // weights_.size() equals sample_points.size()
};


void PreProcess(const PitchExtractionOptions opts,
                const Vector<double> &wave,
                Vector<double> *processed_wave) {
  KALDI_ASSERT(processed_wave != NULL);
  // down-sample and Low-Pass filtering the input wave
  int32 num_samples_in = wave.Dim();
  double dt = opts.samp_freq / opts.resample_freq;
  int32 resampled_len = 1 + static_cast<int>(num_samples_in / dt);
  processed_wave->Resize(resampled_len);  // filtered wave
  std::vector<double> resampled_t(resampled_len);
  for (int32 i = 0; i < resampled_len; i++)
    resampled_t[i] = static_cast<double>(i) / opts.resample_freq;
  Matrix<double> input_wave(1, wave.Dim()), output_wave(1, resampled_len);
  input_wave.CopyRowFromVec(wave, 0);
  ArbitraryResample resample(num_samples_in, opts.samp_freq,
                             opts.lowpass_cutoff,
                             resampled_t, opts.lowpass_filter_width);
  resample.Upsample(input_wave, &output_wave);
  processed_wave->CopyRowFromMat(output_wave, 0);

  // Normalize input signal using rms
  double rms = pow(VecVec((*processed_wave), (*processed_wave)) / processed_wave->Dim(), 0.5);
  if (rms != 0.0)
    (*processed_wave).Scale(1.0 / rms);
}

void Nccf(const Vector<double> &wave,
          int32 start, int32 end,
          int32 nccf_window_size,
          Vector<double> *inner_prod,
          Vector<double> *norm_prod) {
  Vector<double> zero_mean_wave(wave);
  SubVector<double> wave_part(wave, 0, nccf_window_size);
  zero_mean_wave.Add(-wave_part.Sum() / nccf_window_size);  // subtract mean-frame from wave
  double e1, e2, sum;
  SubVector<double> sub_vec1(zero_mean_wave, 0, nccf_window_size);
  e1 = VecVec(sub_vec1, sub_vec1);
  for (int32 lag = start; lag < end; lag++) {
    SubVector<double> sub_vec2(zero_mean_wave, lag, nccf_window_size);
    e2 = VecVec(sub_vec2, sub_vec2);
    sum = VecVec(sub_vec1, sub_vec2);
    (*inner_prod)(lag-start) = sum;
    (*norm_prod)(lag-start) = e1 * e2;
  }
}

void ProcessNccf(const Vector<double> &inner_prod,
                 const Vector<double> &norm_prod,
                 const double &a_fact,
                 int32 start, int32 end,
                 SubVector<double> *autocorr) {
  for (int32 lag = start; lag < end; lag++) {
    if (norm_prod(lag-start) != 0.0)
      (*autocorr)(lag) = inner_prod(lag-start) / pow(norm_prod(lag-start) + a_fact, 0.5);
    KALDI_ASSERT((*autocorr)(lag) < 1.01 && (*autocorr)(lag) > -1.01);
  }
}

void SelectLag(const PitchExtractionOptions &opts,
               int32 *state_num,
               Vector<double> *lags) {
  // choose lags relative to acceptable pitch tolerance
  double min_lag = 1.0 / (1.0 * opts.max_f0),
      max_lag = 1.0 / (1.0 * opts.min_f0);
  double delta_lag = opts.upsample_filter_width/(2.0 * opts.resample_freq);

  int32 lag_size = 1 +
      round(log((max_lag + delta_lag) / (min_lag - delta_lag)) / log(1.0 + opts.delta_pitch));
  lags->Resize(lag_size);

  // we choose sequence of lags which leads to  delta_pitch difference in pitch_space.
  double lag = min_lag;
  int32 count = 0;
  while (lag <= max_lag) {
    (*lags)(count) = lag;
    count++;
    lag = lag * (1 + opts.delta_pitch);
  }
  lags->Resize(count, kCopyData);
  (*state_num) = count;
}

class PitchExtractor {
 public:
  explicit PitchExtractor(const PitchExtractionOptions &opts,
                          const Vector<double> lags,
                          int32 state_num,
                          int32 num_frames) :
      opts_(opts),
      state_num_(state_num),
      num_frames_(num_frames),
      lags_(lags) {
    frames_.resize(num_frames_+1);
    for (int32 i = 0; i < num_frames_+1; i++) {
      frames_[i].local_cost.Resize(state_num_);
      frames_[i].obj_func.Resize(state_num_);
      frames_[i].back_pointers.Resize(state_num_);
    }
  }
  ~PitchExtractor() {}

  void ComputeLocalCost(const Matrix<double> &autocorrelation) {
    Vector<double> correl(state_num_);

    for (int32 i = 1; i < num_frames_ + 1; i++) {
      SubVector<double> frame(autocorrelation.Row(i-1));
      Vector<double> local_cost(state_num_);
      for (int32 j = 0; j < state_num_; j++)
        correl(j) = frame(j);
      // compute the local cost
      frames_[i].local_cost.Add(1.0);
      frames_[i].local_cost.AddVec(-1.0, correl);
      Vector<double> corr_lag_cost(state_num_);
      corr_lag_cost.AddVecVec(opts_.soft_min_f0, correl, lags_, 0);
      frames_[i].local_cost.AddVec(1.0, corr_lag_cost);
    }  // end of loop over frames
  }
  void FastViterbi(const Matrix<double> &correl) {
    ComputeLocalCost(correl);
    double intercost, min_c, this_c;
    int best_b, min_i, max_i;
    BaseFloat delta_pitch_sq = log(1 + opts_.delta_pitch) * log(1 + opts_.delta_pitch);
    // loop over frames
    for (int32 t = 1; t < num_frames_ + 1; t++) {
      // Forward Pass
      for (int32 i = 0; i < state_num_; i++) {
        if ( i == 0 )
          min_i = 0;
        else
          min_i = frames_[t].back_pointers(i-1);
        min_c = std::numeric_limits<double>::infinity();
        best_b = -1;

        for (int32 k = min_i; k <= i; k++) {
          intercost = (i-k) * (i-k) * delta_pitch_sq;
          this_c = frames_[t-1].obj_func(k)+ opts_.penalty_factor * intercost;
          if (this_c < min_c) {
            min_c = this_c;
            best_b = k;
          }
        }
        frames_[t].back_pointers(i) = best_b;
        frames_[t].obj_func(i) = min_c + frames_[t].local_cost(i);
      }
      // Backward Pass
      for (int32 i = state_num_-1; i >= 0; i--) {
        if (i == state_num_-1)
          max_i = state_num_-1;
        else
          max_i = frames_[t].back_pointers(i+1);
        min_c = frames_[t].obj_func(i) - frames_[t].local_cost(i);
        best_b = frames_[t].back_pointers(i);

        for (int32 k = i+1 ; k <= max_i; k++) {
          intercost = (i-k) * (i-k) * delta_pitch_sq;
          this_c = frames_[t-1].obj_func(k)+ opts_.penalty_factor *intercost;
          if (this_c < min_c) {
            min_c = this_c;
            best_b = k;
          }
        }
        frames_[t].back_pointers(i) = best_b;
        frames_[t].obj_func(i) = min_c + frames_[t].local_cost(i);
      }
    }
    // FindBestPath(resampled_nccf_pov);
  }

  void FindBestPath(const Matrix<double> &correlation) {
    // Find the Best path using backpointers
    int32 i = num_frames_;
    int32 best;
    double l_opt;
    frames_[i].obj_func.Min(&best);
    while (i > 0) {
      l_opt = lags_(best);
      frames_[i].truepitch = 1.0 / l_opt;
      frames_[i].pov = correlation(i-1, best);
      best = frames_[i].back_pointers(best);
      i--;
    }
  }
  void GetPitch(Matrix<BaseFloat> *output) {
    output->Resize(num_frames_, 2);
    for (int32 frm = 0; frm < num_frames_; frm++) {
      (*output)(frm, 0) = static_cast<BaseFloat>(frames_[frm + 1].pov);
      (*output)(frm, 1) = static_cast<BaseFloat>(frames_[frm + 1].truepitch);
    } 
  }
 private:
  PitchExtractionOptions opts_;
  int32 state_num_;      // number of states in Viterbi Computation
  int32 num_frames_;     // number of frames in input wave
  Vector<double> lags_;    // all lags used in viterbi
  struct PitchFrame {
    Vector<double> local_cost; 
    Vector<double> obj_func;      // optimal objective function for frame i
    Vector<double> back_pointers;
    double truepitch;             // True pitch
    double pov;                   // probability of voicing
    explicit PitchFrame() {}
  };
  std::vector< PitchFrame > frames_;
};

void Compute(const PitchExtractionOptions &opts,
             const VectorBase<BaseFloat> &wave,
             Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);
  Vector<double> wave2(wave);

  // Preprocess the wave
  Vector<double> processed_wave(wave2.Dim());
  PreProcess(opts, wave2, &processed_wave);
  int32 num_states, rows_out = PitchNumFrames(processed_wave.Dim(), opts);
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << processed_wave.Dim() << ")";
  Vector<double> window;       // windowed waveform.
  double outer_min_lag = 1.0 / (1.0 * opts.max_f0) -
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  double outer_max_lag = 1.0 / (1.0 * opts.min_f0) +
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  int32 num_max_lag = round(outer_max_lag * opts.resample_freq) + 1;
  int32 num_lags = round(opts.resample_freq * outer_max_lag) -
      round(opts.resample_freq *  outer_min_lag) + 1;
  int32 start = round(opts.resample_freq  * outer_min_lag),
      end = round(opts.resample_freq / opts.min_f0) +
      round(opts.lowpass_filter_width / 2);

  Vector<double> lags;
  SelectLag(opts, &num_states, &lags);
  double a_fact_pitch = pow(opts.NccfWindowSize(), 4) * opts.nccf_ballast,
    a_fact_pov = pow(10, -9);
  Matrix<double> nccf_pitch(rows_out, num_max_lag + 1),
      nccf_pov(rows_out, num_max_lag + 1);
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index.
    ExtractFrame(processed_wave, r, opts, &window);
    // compute nccf for pitch extraction
    Vector<double> inner_prod(num_lags), norm_prod(num_lags);
    Nccf(window, start, end, opts.NccfWindowSize(),
         &inner_prod, &norm_prod);
    SubVector<double> nccf_pitch_vec(nccf_pitch.Row(r));
    ProcessNccf(inner_prod, norm_prod, a_fact_pitch, start, end, &(nccf_pitch_vec));
    // compute the Nccf for Probability of voicing estimation
    SubVector<double> nccf_pov_vec(nccf_pov.Row(r));
    ProcessNccf(inner_prod, norm_prod, a_fact_pov, start, end, &(nccf_pov_vec));
  }
  std::vector<double> lag_vec(num_states);
  for (int32 i = 0; i < num_states; i++)
    lag_vec[i] = static_cast<double>(lags(i));
  // upsample the nccf to have better pitch and pov estimation
  ArbitraryResample resample(num_max_lag + 1, opts.resample_freq,
                             opts.upsample_cutoff,
                             lag_vec, opts.upsample_filter_width);
  Matrix<double> resampled_nccf_pitch(rows_out, num_states);
  resample.Upsample(nccf_pitch, &resampled_nccf_pitch);
  Matrix<double> resampled_nccf_pov(rows_out, num_states);
  resample.Upsample(nccf_pov, &resampled_nccf_pov);

  PitchExtractor pitch(opts, lags, num_states, rows_out);
  pitch.FastViterbi(resampled_nccf_pitch);
  pitch.FindBestPath(resampled_nccf_pov);
  output->Resize(rows_out, 2);  // (pov, pitch)
  pitch.GetPitch(output);
}

void ExtractDeltaPitch(const PostProcessPitchOptions &opts,
                       const Vector<BaseFloat> &input,
                       Vector<BaseFloat> *output) {
  int32 num_frames = input.Dim();
  DeltaFeaturesOptions delta_opts;
  delta_opts.order = 1;
  delta_opts.window = opts.delta_window;
  Matrix<BaseFloat> matrix_input(num_frames, 1),
      matrix_output;
  matrix_input.CopyColFromVec(input, 0);
  ComputeDeltas(delta_opts, matrix_input, &matrix_output);
  KALDI_ASSERT(matrix_output.NumRows() == matrix_input.NumRows() &&
               matrix_output.NumCols() == 2);
  output->Resize(num_frames);
  output->CopyColFromMat(matrix_output, 1);

  // Add a small amount of noise to the delta-pitch.. this is to stop peaks
  // appearing in the distribution of delta pitch, that correspond to the
  // discretization interval for log-pitch.
  Vector<BaseFloat> noise(num_frames);
  noise.SetRandn();
  noise.Scale(opts.delta_pitch_noise_stddev);
  output->AddVec(1.0, noise);
}

void PostProcessPitch(const PostProcessPitchOptions &opts,
                      const Matrix<BaseFloat> &input,
                      Matrix<BaseFloat> *output) {
  Matrix<BaseFloat> processed_input = input;
  Vector<BaseFloat> pov(input.NumRows()), pitch(input.NumRows());
  pov.CopyColFromMat(input, 0);
  pitch.CopyColFromMat(input, 1);
  bool apply_sigmoid = true;
  int nonlinearity = 2;  // use new nonlinearity
  TakeLogOfPitch(&processed_input);
  ProcessPovFeatures(&processed_input, nonlinearity, apply_sigmoid);
  Matrix<BaseFloat> processed_output(processed_input);
  WeightedMwn(opts.normalization_window_size, 
              processed_input, &processed_output);
  processed_output.CopyColFromVec(pov, 0);
  apply_sigmoid = false;
  ProcessPovFeatures(&processed_output, opts.pov_nonlinearity, apply_sigmoid);
  pov.CopyColFromMat(processed_output, 0);
  pov.Scale(opts.pov_scale);
  pitch.CopyColFromMat(processed_output, 1);
  pitch.Scale(opts.pitch_scale);

  if (opts.add_delta_pitch) {
    Vector<BaseFloat> delta_pitch, log_pitch(input.NumRows());
    log_pitch.CopyColFromMat(processed_input, 1);
    ExtractDeltaPitch(opts, log_pitch, &delta_pitch);
    delta_pitch.Scale(opts.delta_pitch_scale);
    output->Resize(input.NumRows(), 3);
    output->SetZero();
    output->CopyColFromVec(pov, 0);
    output->CopyColFromVec(pitch, 1);
    output->CopyColFromVec(delta_pitch, 2);
  } else {
    output->Resize(input.NumRows(), 2);
    output->SetZero();
    output->CopyColFromVec(pov, 0);
    output->CopyColFromVec(pitch, 1);
  }
}

}  // namespace kaldi
