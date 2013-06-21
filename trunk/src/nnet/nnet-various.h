// nnet/nnet-various.h

// Copyright 2012 Karel Vesely

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


#ifndef KALDI_NNET_NNET_VARIOUS_H_
#define KALDI_NNET_NNET_VARIOUS_H_

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {




/**
 * Splices the time context of the input features
 * in N, out k*N, FrameOffset o_1,o_2,...,o_k
 * FrameOffset example 11frames: -5 -4 -3 -2 -1 0 1 2 3 4 5
 */
class Splice : public Component {
 public:
  Splice(int32 dim_in, int32 dim_out, Nnet *nnet)
    : Component(dim_in, dim_out, nnet)
  { }
  ~Splice()
  { }

  ComponentType GetType() const { 
    return kSplice; 
  }

  void ReadData(std::istream &is, bool binary) {
    //read double vector
    Vector<double> vec_d;
    vec_d.Read(is, binary);
    //convert to int vector
    std::vector<int32> vec_i(vec_d.Dim());
    for(int32 i=0; i<vec_d.Dim(); i++) {
      vec_i[i] = round(vec_d(i));
    }
    //push to GPU
    frame_offsets_.CopyFromVec(vec_i); 
  }

  void WriteData(std::ostream &os, bool binary) const {
    std::vector<int32> vec_i;
    frame_offsets_.CopyToVec(&vec_i);
    Vector<double> vec_d(vec_i.size());
    for(int32 i=0; i<vec_d.Dim(); i++) {
      vec_d(i) = vec_i[i];
    }
    vec_d.Write(os, binary);
  }
   
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    cu::Splice(in, frame_offsets_, out); 
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    KALDI_ERR << __func__ << "Not implemented!";
  }

 protected:
  CuStlVector<int32> frame_offsets_;
};



/**
 * Rearrange the matrix columns according to the indices in copy_from_indices_
 */
class Copy : public Component {
 public:
  Copy(int32 dim_in, int32 dim_out, Nnet *nnet)
    : Component(dim_in, dim_out, nnet)
  { }
  ~Copy()
  { }

  ComponentType GetType() const { 
    return kCopy; 
  }

  void ReadData(std::istream &is, bool binary) { 
    //read double vector
    Vector<double> vec_d;
    vec_d.Read(is, binary);
    //subtract 1
    vec_d.Add(-1.0);
    //convert to int vector
    std::vector<int32> vec_i(vec_d.Dim());
    for(int32 i=0; i<vec_d.Dim(); i++) {
      vec_i[i] = round(vec_d(i));
    }
    //push to GPU
    copy_from_indices_.CopyFromVec(vec_i); 
  }

  void WriteData(std::ostream &os, bool binary) const { 
    std::vector<int32> vec_i;
    copy_from_indices_.CopyToVec(&vec_i);
    Vector<double> vec_d(vec_i.size());
    for(int32 i=0; i<vec_d.Dim(); i++) {
      vec_d(i) = vec_i[i];
    }
    vec_d.Add(1.0);
    vec_d.Write(os, binary);
  }
   
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) { 
    cu::Copy(in,copy_from_indices_,out); 
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    KALDI_ERR << __func__ << "Not implemented!";
  }

 protected:
  CuStlVector<int32> copy_from_indices_;
};



/**
 * Adds shift to all the lines of the matrix
 * (can be used for global mean normalization)
 */
class AddShift : public Component {
 public:
  AddShift(int32 dim_in, int32 dim_out, Nnet *nnet)
    : Component(dim_in, dim_out, nnet), shift_data_(dim_in)
  { }
  ~AddShift()
  { }

  ComponentType GetType() const { 
    return kAddShift; 
  }

  void ReadData(std::istream &is, bool binary) { 
    //read the shift data
    shift_data_.Read(is, binary);
  }

  void WriteData(std::ostream &os, bool binary) const { 
    shift_data_.Write(os, binary);
  }
   
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) { 
    out->CopyFromMat(in);
    //add the shift
    out->AddVecToRows(1.0, shift_data_, 1.0);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    //derivative of additive constant is zero...
    in_diff->CopyFromMat(out_diff);
  }

  //Data accessors
  const CuVector<BaseFloat>& GetShiftVec() {
    return shift_data_;
  }

  void SetShiftVec(const CuVector<BaseFloat>& shift_data) {
    KALDI_ASSERT(shift_data.Dim() == shift_data_.Dim());
    shift_data_.CopyFromVec(shift_data);
  }

 protected:
  CuVector<BaseFloat> shift_data_;
};



/**
 * Rescale the data column-wise by a vector
 * (can be used for global variance normalization)
 */
class Rescale : public Component {
 public:
  Rescale(int32 dim_in, int32 dim_out, Nnet *nnet)
    : Component(dim_in, dim_out, nnet), scale_data_(dim_in)
  { }
  ~Rescale()
  { }

  ComponentType GetType() const { 
    return kRescale; 
  }

  void ReadData(std::istream &is, bool binary) { 
    //read the shift data
    scale_data_.Read(is, binary);
  }

  void WriteData(std::ostream &os, bool binary) const { 
    scale_data_.Write(os, binary);
  }
   
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) { 
    out->CopyFromMat(in);
    //rescale the data
    out->MulColsVec(scale_data_);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    //derivative gets also scaled by the scale_data_
    in_diff->MulColsVec(scale_data_);
  }

  //Data accessors
  const CuVector<BaseFloat>& GetScaleVec() {
    return scale_data_;
  }

  void SetScaleVec(const CuVector<BaseFloat>& scale_data) {
    KALDI_ASSERT(scale_data.Dim() == scale_data_.Dim());
    scale_data_.CopyFromVec(scale_data);
  }

 protected:
  CuVector<BaseFloat> scale_data_;
};






} // namespace nnet1
} // namespace kaldi

#endif // KALDI_NNET_NNET_VARIOUS_H_
