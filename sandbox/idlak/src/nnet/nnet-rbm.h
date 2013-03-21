// nnet/nnet-rbm.h

// Copyright 2012  Karel Vesely

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


#ifndef KALDI_NNET_RBM_H
#define KALDI_NNET_RBM_H


#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

class RbmBase : public UpdatableComponent {
 public:
  typedef enum {
    BERNOULLI,
    GAUSSIAN
  } RbmNodeType;
 
  RbmBase(int32 dim_in, int32 dim_out, Nnet *nnet) 
   : UpdatableComponent(dim_in, dim_out, nnet)
  { }
  
  /*Is included in Component:: itf
  virtual void Propagate(
    const CuMatrix<BaseFloat> &vis_probs, 
    CuMatrix<BaseFloat> *hid_probs
  ) = 0;
  */

  virtual void Reconstruct(
    const CuMatrix<BaseFloat> &hid_state, 
    CuMatrix<BaseFloat> *vis_probs
  ) = 0;
  virtual void RbmUpdate(
    const CuMatrix<BaseFloat> &pos_vis, 
    const CuMatrix<BaseFloat> &pos_hid, 
    const CuMatrix<BaseFloat> &neg_vis, 
    const CuMatrix<BaseFloat> &neg_hid
  ) = 0;

  virtual RbmNodeType VisType() const = 0;
  virtual RbmNodeType HidType() const = 0;

  virtual void WriteAsNnet(std::ostream& os, bool binary) const = 0;
};



class Rbm : public RbmBase {
 public:
  Rbm(int32 dim_in, int32 dim_out, Nnet *nnet) 
   : RbmBase(dim_in, dim_out, nnet)
  { } 
  ~Rbm()
  { }  
  
  ComponentType GetType() const {
    return kRbm;
  }

  void ReadData(std::istream &is, bool binary) {
    std::string vis_node_type, hid_node_type;
    ReadToken(is, binary, &vis_node_type);
    ReadToken(is, binary, &hid_node_type);
    
    if(vis_node_type == "bern") {
      vis_type_ = RbmBase::BERNOULLI;
    } else if(vis_node_type == "gauss") {
      vis_type_ = RbmBase::GAUSSIAN;
    }
    if(hid_node_type == "bern") {
      hid_type_ = RbmBase::BERNOULLI;
    } else if(hid_node_type == "gauss") {
      hid_type_ = RbmBase::GAUSSIAN;
    }

    vis_hid_.Read(is, binary);
    vis_bias_.Read(is, binary);
    hid_bias_.Read(is, binary);

    KALDI_ASSERT(vis_hid_.NumRows() == output_dim_);
    KALDI_ASSERT(vis_hid_.NumCols() == input_dim_);
    KALDI_ASSERT(vis_bias_.Dim() == input_dim_);
    KALDI_ASSERT(hid_bias_.Dim() == output_dim_);
  }
  
  void WriteData(std::ostream &os, bool binary) const {
    switch (vis_type_) {
      case BERNOULLI : WriteToken(os,binary,"bern"); break;
      case GAUSSIAN  : WriteToken(os,binary,"gauss"); break;
      default : KALDI_ERR << "Unknown type " << vis_type_;
    }
    switch (hid_type_) {
      case BERNOULLI : WriteToken(os,binary,"bern"); break;
      case GAUSSIAN  : WriteToken(os,binary,"gauss"); break;
      default : KALDI_ERR << "Unknown type " << hid_type_;
    }
    vis_hid_.Write(os, binary);
    vis_bias_.Write(os, binary);
    hid_bias_.Write(os, binary);
  }


  // UpdatableComponent API
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, hid_bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, vis_hid_, kTrans, 1.0);
    // optionally apply sigmoid
    if (hid_type_ == RbmBase::BERNOULLI) {
      out->Sigmoid(*out);
    }
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    KALDI_ERR << "Cannot backpropagate through RBM!"
              << "Better convert it to <affinetransform> and <sigmoid>";
  }
  virtual void Update(const CuMatrix<BaseFloat> &input,
                      const CuMatrix<BaseFloat> &diff) {
    KALDI_ERR << "Cannot update RBM by backprop!"
              << "Better convert it to <affinetransform> and <sigmoid>";
  }

  // RBM training API
  void Reconstruct(const CuMatrix<BaseFloat> &hid_state, CuMatrix<BaseFloat> *vis_probs) {
    // check the dim
    if (output_dim_ != hid_state.NumCols()) {
      KALDI_ERR << "Nonmatching dims, component:" << output_dim_ << " data:" << hid_state.NumCols();
    }
    // optionally allocate buffer
    if (input_dim_ != vis_probs->NumCols() || hid_state.NumRows() != vis_probs->NumRows()) {
      vis_probs->Resize(hid_state.NumRows(), input_dim_);
    }

    // precopy bias
    vis_probs->AddVecToRows(1.0, vis_bias_, 0.0);
    // multiply by weights
    vis_probs->AddMatMat(1.0, hid_state, kNoTrans, vis_hid_, kNoTrans, 1.0);
    // optionally apply sigmoid
    if (vis_type_ == RbmBase::BERNOULLI) {
      vis_probs->Sigmoid(*vis_probs);
    }
  }
  
  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid, const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid) {

    assert(pos_vis.NumRows() == pos_hid.NumRows() &&
           pos_vis.NumRows() == neg_vis.NumRows() &&
           pos_vis.NumRows() == neg_hid.NumRows() &&
           pos_vis.NumCols() == neg_vis.NumCols() &&
           pos_hid.NumCols() == neg_hid.NumCols() &&
           pos_vis.NumCols() == input_dim_ &&
           pos_hid.NumCols() == output_dim_);

    //lazy initialization of buffers
    if ( vis_hid_corr_.NumRows() != vis_hid_.NumRows() ||
         vis_hid_corr_.NumCols() != vis_hid_.NumCols() ||
         vis_bias_corr_.Dim()    != vis_bias_.Dim()    ||
         hid_bias_corr_.Dim()    != hid_bias_.Dim()     ){
      vis_hid_corr_.Resize(vis_hid_.NumRows(),vis_hid_.NumCols(),kSetZero);
      //vis_bias_corr_.Resize(vis_bias_.Dim(),kSetZero);
      //hid_bias_corr_.Resize(hid_bias_.Dim(),kSetZero);
      vis_bias_corr_.Resize(vis_bias_.Dim());
      hid_bias_corr_.Resize(hid_bias_.Dim());
    }

    //TODO: detect divergence condition of Gaussian-Bernoulli unit! (scale down vis_hid_corr_?)
    // limit the reconstruction variace by limiting the weight size???!!!
    //
    // Or detect the divergence by getting standard deviation of a)input minibatch b)reconstruction
    // When ratio b)/a) larger than N (2 or 10...):
    // 1. scale down the weights and biases by b)/a) ratio of stddev 
    // 2. shrink learning rate by 0.9x
    // 3. reset the momentum buffer  
    // display warning! in later stage the training can return back to higher learning rate
    // 
    if (vis_type_ == RbmBase::GAUSSIAN) {
      //get the standard deviations of pos_vis and neg_vis data

      //pos_vis
      CuMatrix<BaseFloat> pos_vis_pow2(pos_vis);
      pos_vis_pow2.MulElements(pos_vis);
      CuVector<BaseFloat> pos_vis_second(pos_vis.NumCols());
      pos_vis_second.AddRowSumMat(1.0,pos_vis_pow2,0.0);
      CuVector<BaseFloat> pos_vis_mean(pos_vis.NumCols());
      pos_vis_mean.AddRowSumMat(1.0/pos_vis.NumRows(),pos_vis,0.0);

      Vector<BaseFloat> pos_vis_second_h(pos_vis_second.Dim());
      pos_vis_second.CopyToVec(&pos_vis_second_h);
      Vector<BaseFloat> pos_vis_mean_h(pos_vis_mean.Dim());
      pos_vis_mean.CopyToVec(&pos_vis_mean_h);
      
      Vector<BaseFloat> pos_vis_stddev(pos_vis_mean_h);
      pos_vis_stddev.MulElements(pos_vis_mean_h);
      pos_vis_stddev.Scale(-1.0);
      pos_vis_stddev.AddVec(1.0/pos_vis.NumRows(),pos_vis_second_h);
      pos_vis_stddev.ApplyPow(0.5);

      //neg_vis
      CuMatrix<BaseFloat> neg_vis_pow2(neg_vis);
      neg_vis_pow2.MulElements(neg_vis);
      CuVector<BaseFloat> neg_vis_second(neg_vis.NumCols());
      neg_vis_second.AddRowSumMat(1.0,neg_vis_pow2,0.0);
      CuVector<BaseFloat> neg_vis_mean(neg_vis.NumCols());
      neg_vis_mean.AddRowSumMat(1.0/neg_vis.NumRows(),neg_vis,0.0);

      Vector<BaseFloat> neg_vis_second_h(neg_vis_second.Dim());
      neg_vis_second.CopyToVec(&neg_vis_second_h);
      Vector<BaseFloat> neg_vis_mean_h(neg_vis_mean.Dim());
      neg_vis_mean.CopyToVec(&neg_vis_mean_h);
      
      Vector<BaseFloat> neg_vis_stddev(neg_vis_mean_h);
      neg_vis_stddev.MulElements(neg_vis_mean_h);
      neg_vis_stddev.Scale(-1.0);
      neg_vis_stddev.AddVec(1.0/neg_vis.NumRows(),neg_vis_second_h);
      /* set negtive values to zero before the square root */
      for (int32 i=0; i<neg_vis_stddev.Dim(); i++) {
        if(neg_vis_stddev(i) < 0.0) { 
          KALDI_WARN << "Forcing the variance to be non-negative! (set to zero)" << neg_vis_stddev(i);
          neg_vis_stddev(i) = 0.0;
        }
      }
      neg_vis_stddev.ApplyPow(0.5);

      //monitor the standard deviation discrepancy between pos_vis and neg_vis
      if (pos_vis_stddev.Sum() * 2 < neg_vis_stddev.Sum()) {
        //scale-down the weights and biases
        BaseFloat scale = pos_vis_stddev.Sum() / neg_vis_stddev.Sum();
        vis_hid_.Scale(scale);
        vis_bias_.Scale(scale);
        hid_bias_.Scale(scale);
        //reduce the learning rate           
        learn_rate_ *= 0.9;
        //reset the momentum buffers
        vis_hid_corr_.SetZero();
        vis_bias_corr_.SetZero();
        hid_bias_corr_.SetZero();

        KALDI_WARN << "Discrepancy between pos_hid and neg_hid varainces, "
                   << "danger of weight explosion. a) Reducing weights with scale " << scale
                   << " b) Lowering learning rate to " << learn_rate_
                   << " [pos_vis_stddev(~1.0):" << pos_vis_stddev.Sum()/pos_vis.NumCols()
                   << ",neg_vis_stddev:" << neg_vis_stddev.Sum()/neg_vis.NumCols() << "]";
        return;           
      }
    }
    

    //  UPDATE vishid matrix
    //  
    //  vishidinc = momentum*vishidinc + ...
    //              epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    //
    //  vishidinc[t] = -(epsilonw/numcases)*negprods + momentum*vishidinc[t-1]
    //                 +(epsilonw/numcases)*posprods
    //                 -(epsilonw*weightcost)*vishid[t-1]
    //
    BaseFloat N = static_cast<BaseFloat>(pos_vis.NumRows());
    vis_hid_corr_.AddMatMat(-learn_rate_/N, neg_hid, kTrans, neg_vis, kNoTrans, momentum_);
    vis_hid_corr_.AddMatMat(+learn_rate_/N, pos_hid, kTrans, pos_vis, kNoTrans, 1.0);
    vis_hid_corr_.AddMat(-learn_rate_*l2_penalty_, vis_hid_, 1.0);
    vis_hid_.AddMat(1.0, vis_hid_corr_, 1.0);

    //  UPDATE visbias vector
    //
    //  visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    //
    vis_bias_corr_.AddRowSumMat(-learn_rate_/N, neg_vis, momentum_);
    vis_bias_corr_.AddRowSumMat(+learn_rate_/N, pos_vis, 1.0);
    vis_bias_.AddVec(1.0, vis_bias_corr_, 1.0);
    
    //  UPDATE hidbias vector
    //
    // hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    //
    hid_bias_corr_.AddRowSumMat(-learn_rate_/N, neg_hid, momentum_);
    hid_bias_corr_.AddRowSumMat(+learn_rate_/N, pos_hid, 1.0);
    hid_bias_.AddVec(1.0, hid_bias_corr_, 1.0);
  }



  RbmNodeType VisType() const { 
    return vis_type_; 
  }

  RbmNodeType HidType() const { 
    return hid_type_; 
  }

  void WriteAsNnet(std::ostream& os, bool binary) const {
    //header
    WriteToken(os,binary,Component::TypeToMarker(Component::kAffineTransform));
    WriteBasicType(os,binary,OutputDim());
    WriteBasicType(os,binary,InputDim());
    if(!binary) os << "\n";
    //data
    vis_hid_.Write(os,binary);
    hid_bias_.Write(os,binary);
    //optionally sigmoid activation
    if(HidType() == BERNOULLI) {
      WriteToken(os,binary,Component::TypeToMarker(Component::kSigmoid));
      WriteBasicType(os,binary,OutputDim());
      WriteBasicType(os,binary,OutputDim());
    }
    if(!binary) os << "\n";
  }

protected:
  CuMatrix<BaseFloat> vis_hid_;        ///< Matrix with neuron weights
  CuVector<BaseFloat> vis_bias_;       ///< Vector with biases
  CuVector<BaseFloat> hid_bias_;       ///< Vector with biases

  CuMatrix<BaseFloat> vis_hid_corr_;   ///< Matrix for linearity updates
  CuVector<BaseFloat> vis_bias_corr_;  ///< Vector for bias updates
  CuVector<BaseFloat> hid_bias_corr_;  ///< Vector for bias updates

  RbmNodeType vis_type_;
  RbmNodeType hid_type_;

};



} // namespace

#endif
