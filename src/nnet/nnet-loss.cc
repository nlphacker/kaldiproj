// nnet/nnet-loss.cc

// Copyright 2011  Karel Vesely

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

#include "nnet/nnet-loss.h"

#include "cudamatrix/cu-math.h"

#include <sstream>
#include <iterator>

namespace kaldi {


void Xent::Eval(const CuMatrix<BaseFloat> &net_out, const CuMatrix<BaseFloat> &target, CuMatrix<BaseFloat> *diff) {
  
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  diff->Resize(net_out.NumRows(), net_out.NumCols());

  // compute derivative wrt. activations of last layer of neurons
  diff->CopyFromMat(net_out);
  diff->AddMat(-1.0, target);

  // we'll not produce per-frame classification accuracy for soft labels
  correct_ = -1;

  // :TODO: reimplement when needed
  // compute xentropy (ON CPU)
  Matrix<BaseFloat> target_host, net_out_host;
  target.CopyToMat(&target_host);
  net_out.CopyToMat(&net_out_host);
  BaseFloat val;
  for(int32 r=0; r<net_out.NumRows(); r++) {
    for(int32 c=0; c<net_out.NumCols(); c++) {
      val = -target_host(r, c)*log(net_out_host(r, c));
      if (KALDI_ISINF(val)) val = 1e10;
      loss_ += val;
    }
  }

  frames_ += net_out.NumRows();
}


void Xent::EvalVec(const CuMatrix<BaseFloat> &net_out, const std::vector<int32> &target, CuMatrix<BaseFloat> *diff) {
  // evaluate the frame-level classification
  int32 correct=0;
  cu::FindRowMaxId(net_out, &max_id_);
  max_id_.CopyToVec(&max_id_host_);
  KALDI_ASSERT(max_id_host_.size() == target.size());
  for(int32 i=0; i<target.size(); i++) {
    if (target[i] == max_id_host_[i]) correct++;
  }
  
  // get the xentropy and global error 
  target_device_.CopyFromVec(target);
  if(&net_out != diff) { //<allow no-copy speedup
    diff->CopyFromMat(net_out);
  }
  cu::DiffXent(target_device_, diff, &log_post_tgt_);
  // Now we have derivative of Xentropy in diff,
  // it's computed  as (net_out - target);
  
  // The xentropy value is computed as -sum(log(net_out)) 
  // by using target indices as a mask, 
  // the masked logposteriors are now in log_post_tgt_;
  log_post_tgt_.CopyToVec(&log_post_tgt_host_);
  loss_    -= log_post_tgt_host_.Sum();
  
  // accumulate error quantites
  frames_  += net_out.NumRows();
  correct_ += correct;
   
}


std::string Xent::Report() {
  std::ostringstream oss;
  oss << "Xent:" << loss_ << " frames:" << frames_ 
      << " err/frm:" << loss_/frames_;
  if (correct_ >= 0.0) {
    oss << "\nFRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<";
  }
  return oss.str(); 
}




void Mse::Eval(const CuMatrix<BaseFloat> &net_out, const CuMatrix<BaseFloat> &target, CuMatrix<BaseFloat> *diff) {
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  diff->Resize(net_out.NumRows(), net_out.NumCols());

  // compute derivative w.r.t. neural nerwork outputs
  diff->CopyFromMat(net_out);
  diff->AddMat(-1.0, target);

  // :TODO: reimplement when needed
  // compute mean square error (ON CPU)
  Matrix<BaseFloat> target_host, net_out_host;
  target.CopyToMat(&target_host);
  net_out.CopyToMat(&net_out_host);
  BaseFloat val;
  double loss=0.0;
  for(int32 r=0; r<net_out.NumRows(); r++) {
    for(int32 c=0; c<net_out.NumCols(); c++) {
      val = target_host(r, c) - net_out_host(r, c);
      loss += val*val;
    }
  }
  // :TODO:
  
  frames_ += net_out.NumRows();
  loss_ += loss;
}


std::string Mse::Report() {
  std::ostringstream oss;
  oss << "Mse:" << loss_ << " frames:" << frames_
      << " err/frm:" << loss_/frames_ 
      << std::endl;
  return oss.str();
}




void MseProgress::Eval(const CuMatrix<BaseFloat>& net_out, const CuMatrix<BaseFloat>& target, CuMatrix<BaseFloat>* diff) {
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  diff->Resize(net_out.NumRows(),net_out.NumCols());

  //compute derivative w.r.t. neural nerwork outputs
  diff->CopyFromMat(net_out);
  diff->AddMat(-1.0,target);

  //:TODO: reimplement when needed
  //compute mean square error (ON CPU)
  Matrix<BaseFloat> target_host, net_out_host;
  target.CopyToMat(&target_host);
  net_out.CopyToMat(&net_out_host);
  BaseFloat val;
  double loss=0.0;
  for(int32 r=0; r<net_out.NumRows(); r++) {
    for(int32 c=0; c<net_out.NumCols(); c++) {
      val = target_host(r,c) - net_out_host(r,c);
      loss += val*val;
    }
  }
  //:TODO:
  
  frames_progress_ += net_out.NumRows();
  loss_progress_ += loss;

  //copmpute partial progress statistics
  if(frames_progress_ > progress_step_) {
    float loss_of_step = loss_progress_/frames_progress_;
    loss_vec_.push_back(loss_of_step);
    frames_ += frames_progress_; 
    loss_ += loss_progress_;
    KALDI_LOG << "Progress chunk #" << ++progress_ctr_ << " mse:" << loss_of_step << " [last " << frames_progress_/100/3600 << "h/" << frames_/100/3600 << "h]";
    frames_progress_ = 0;
    loss_progress_ = 0;
  }
}


std::string MseProgress::Report() {
  std::ostringstream oss;
  oss << "Mse:" << loss_ << " frames:" << frames_
      << " err/frm:" << loss_/frames_ 
      << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(),loss_vec_.end(),std::ostream_iterator<float>(oss," "));
  oss << "]" << std::endl;

  return oss.str();
}


} // namespace
