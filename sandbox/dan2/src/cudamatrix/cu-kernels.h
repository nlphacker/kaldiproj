// cudamatrix/cu-kernels.h

// Copyright 2009-2012  Karel Vesely
//                2013  Ehsan Variani
//                2014  Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_KERNELS_H_
#define KALDI_CUDAMATRIX_CU_KERNELS_H_

#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "cudamatrix/cu-kernels-ansi.h"

/*
 * In this file are C++ templated wrappers 
 * of the ANSI-C CUDA kernels
 */

namespace kaldi {



/*********************************************************
 * base templates
 */

/*
 * CuMatrix
 */
template<typename Real> inline void cuda_ammdm_elements(dim3 Gr, dim3 Bl, Real alpha, Real* mat, const Real* A, const Real* B, const Real* C, Real beta, MatrixDim d) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_copy_from_tp_trans(int Gr, int Bl, Real* A, const Real* B, MatrixDim dmat) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_copy_from_tp(int Gr, int Bl, Real* A, const Real* B, MatrixDim dmat) { KALDI_ERR << __func__ << "Not implementde!"; }
template<typename Real> inline void cuda_trace_sp_sp_fd(int Gr, int Bl, const float* A, const Real* B, float* value, int dim) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_trace_sp_sp_df(int Gr, int Bl, const double* A, const Real* B, double* value, int dim) { KALDI_ERR << __func__ << "Not implemented!"; } 
template<typename Real> inline void cuda_copy_from_mat_fd(dim3 Gr, dim3 Bl, float* mat_out, const Real* mat_in, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_copy_from_mat_df(dim3 Gr, dim3 Bl, double* mat_out, const Real* mat_in, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_copy_from_mat_fd_trans(dim3 Gr, dim3 Bl, float* mat_out, const Real* mat_in, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_copy_from_mat_df_trans(dim3 Gr, dim3 Bl, double* mat_out, const Real* mat_in, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_copy_col_from_vec(int Gr, int Bl, Real* mat, const Real* v, int col, MatrixDim d) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_apply_exp(dim3 Gr, dim3 Bl, Real* mat, MatrixDim d) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_sum(dim3 Gr, dim3 Bl, Real* mat, Real* value, MatrixDim d) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_apply_pow(dim3 Gr, dim3 Bl, Real* mat, Real power, MatrixDim dim) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_apply_floor(dim3 Gr, dim3 Bl, Real* mat, Real floor_val, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_trace(int Gr, int Bl, Real* mat, Real* value, int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_set_diag(int Gr, int Bl, Real* mat, Real value, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_set_diag_packed(int Gr, int Bl, Real* mat, Real value, int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_diag_packed(int Gr, int Bl, Real* mat, Real value, int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_set_const(dim3 Gr, dim3 Bl, Real *mat, Real value, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_set_zero_above_diag(dim3 Gr, dim3 Bl, Real* mat, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add(dim3 Gr, dim3 Bl, Real *mat, Real value, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_vec2(dim3 Gr, dim3 Bl, Real *mat, const Real *vec, const Real alpha, int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_scale_diag(int Gr, int Bl, Real* mat, Real value, int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_scale(dim3 Gr, dim3 Bl, Real *mat, Real value, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_apply_log(dim3 Gr, dim3 Bl, Real *mat, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_mul_elements(dim3 Gr, dim3 Bl, Real *mat, const Real *A, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_mul_cols_vec(dim3 Gr, dim3 Bl, Real *mat, const Real *scale, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_mul_rows_vec(dim3 Gr, dim3 Bl, Real *mat, const Real *scale, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_div_rows_vec(dim3 Gr, dim3 Bl, Real *mat, const Real *vec_div, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_mat(dim3 Gr, dim3 Bl, Real alpha, const Real *A, Real beta, Real *dst, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_vec_to_cols(dim3 Gr, dim3 Bl, Real alpha, const Real *col, Real beta, Real *dst, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_vec_to_rows(dim3 Gr, dim3 Bl, Real alpha, const Real *row, Real beta, Real *dst, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
 
/*
 * CuVector
 */
template<typename Real> inline void cuda_set_bias_params(int Gr, int Bl, Real* v, const Real* a, Real param_1, Real param_2, Real param_3, int* flag, int dim) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_copy_from_vec_df(int Gr, int Bl, double* v_out, const Real* v_in, int dim) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_copy_from_vec_fd(int Gr, int Bl, float* v_out, const Real* v_in, int dim) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_vec_mul_elements(int Gr, int Bl, Real* v, const Real* a, int dim) { KALDI_ERR << __func__ << "Not implemented!"; }
template<typename Real> inline void cuda_vec_soft_max(int Gr, int Bl, Real* x, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_min(int Gr, int Bl, const Real* v, Real* value, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_trace_mat_mat_trans(int Gr, int Bl, const Real* A, const Real* B, MatrixDim dA, MatrixDim dB, Real* value) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_trace_mat_mat(int Gr, int Bl, const Real* A, const Real* B, MatrixDim dA, MatrixDim dB, Real* value) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_add_diag_mat_trans(int Gr, int Bl, Real alpha, Real* v, const Real* mat, Real beta, MatrixDim dmat, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_add_diag_mat(int Gr, int Bl, Real alpha, Real* v, const Real* mat, Real beta, MatrixDim dmat, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_add_vec_vec(int Gr, int Bl, Real alpha, Real* v, const Real* x, const Real* y, Real beta, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_copy_col_from_mat(int Gr, int Bl, Real* v, int col, const Real* mat, MatrixDim dmat, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_copy_col_from_mat_df(int Gr, int Bl, double* v, int col, const Real* mat, MatrixDim dmat, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_copy_col_from_mat_fd(int Gr, int Bl, float* v, int col, const Real* mat, MatrixDim dmat, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_vec_sum(int Gr, int Bl, Real* v, Real* value, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_vec_apply_floor(int Gr, int Bl, Real* v, Real floor_val, int* num, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_vec_apply_exp(int Gr, int Bl, Real* v, int dim) { KALDI_ERR << __func__ << " Not implemented! "; }
template<typename Real> inline void cuda_vec_apply_log(int Gr, int Bl, Real* v, Real* flag, int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_row_sum_mat(dim3 Gr, dim3 Bl, const Real *mat, Real *vec_sum, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_add_col_sum_mat(dim3 Gr, dim3 Bl, const Real *mat, Real *vec_sum, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_invert_elements(dim3 Gr, dim3 Bl, Real *data, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }

template<typename Real> inline void cuda_sigmoid(dim3 Gr, dim3 Bl, Real *y, const Real *x, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_diff_sigmoid(dim3 Gr, dim3 Bl, Real *eout, const Real *e, const Real *y, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_tanh(dim3 Gr, dim3 Bl, Real *y, const Real *x, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_diff_tanh(dim3 Gr, dim3 Bl, Real *eout, const Real *e, const Real *y, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_softmax(size_t Gr, size_t Bl, Real *y, const Real *x, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_softmax_part(dim3 Gr, dim3 Bl, const Real *X, const int32_cuda *vec_ids, Real* Y, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }

template<typename Real> inline void cuda_regularize_l1(dim3 Gr, dim3 Bl, Real *wei, Real *grad, Real l1, Real lr, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_find_row_max_id(dim3 Gr, dim3 Bl, const Real *mat, Real *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_diff_xent(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, Real *mat_net_out, Real *vec_log_post, MatrixDim d) { KALDI_ERR << __func__ << " Not implemented!"; }

template<typename Real> inline void cuda_randomize(dim3 Gr, dim3 Bl, Real *y, const Real *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_splice(dim3 Gr, dim3 Bl, Real *y, const Real *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_one(int Gr,int Bl,Real* x,int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_copy(dim3 Gr, dim3 Bl, Real *y, const Real *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_copy_diag(int Gr, int Bl, Real* y, const Real* x, int dim) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_copy_from_sp(int Gr, int Bl, const Real* x, Real* y, int d_in, MatrixDim d_out) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_take_lower(dim3 Gr, dim3 Bl, const Real* x, Real* y, MatrixDim d_in, int d_out) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_take_upper(dim3 Gr, dim3 Bl, const Real* x, Real* y, MatrixDim d_in, int d_out) { KALDI_ERR << __func__ << " Not implemented!"; }
template<typename Real> inline void cuda_take_mean(dim3 Gr, dim3 Bl, const Real* x, Real* y, MatrixDim d_in, int d_out) { KALDI_ERR << __func__ << " Not implemented!"; }
/*********************************************************
 * float specializations
 */

/*
 * CuMatrix 
 */
template<> inline void cuda_ammdm_elements<float>(dim3 Gr, dim3 Bl, float alpha, float* mat, const float* A, const float* B, const float* C, float beta, MatrixDim d) { cudaF_ammdm_elements(Gr,Bl,alpha,mat,A,B,C,beta,d); } 
template<> inline void cuda_copy_from_tp_trans<float>(int Gr, int Bl, float* A, const float* B, MatrixDim dmat) { cudaF_copy_from_tp_trans(Gr,Bl,A,B,dmat); }
template<> inline void cuda_copy_from_tp<float>(int Gr, int Bl, float* A, const float* B, MatrixDim dmat) { cudaF_copy_from_tp(Gr,Bl,A,B,dmat); }
template<> inline void cuda_trace_sp_sp_fd<float>(int Gr, int Bl, const float* A, const float* B, float* value, int dim) { cudaF_trace_sp_sp_fd(Gr,Bl,A,B,value,dim); }
template<> inline void cuda_trace_sp_sp_df<float>(int Gr, int Bl, const double* A, const float* B, double* value, int dim) { cudaF_trace_sp_sp_df(Gr,Bl,A,B,value,dim); }
template<> inline void cuda_copy_from_mat_fd<float>(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) { cudaF_copy_from_mat_fd(Gr,Bl,mat_out,mat_in,d_out,d_in); }
template<> inline void cuda_copy_from_mat_df<float>(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) { cudaF_copy_from_mat_df(Gr,Bl,mat_out,mat_in,d_out,d_in); }
template<> inline void cuda_copy_from_mat_fd_trans<float>(dim3 Gr, dim3 Bl, float* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) { cudaF_copy_from_mat_fd_trans(Gr,Bl,mat_out,mat_in,d_out,d_in); }
template<> inline void cuda_copy_from_mat_df_trans<float>(dim3 Gr, dim3 Bl, double* mat_out, const float* mat_in, MatrixDim d_out, MatrixDim d_in) { cudaF_copy_from_mat_df_trans(Gr,Bl,mat_out,mat_in,d_out,d_in); }
template<> inline void cuda_copy_col_from_vec<float>(int Gr, int Bl, float* mat, const float* v, int col, MatrixDim d) { cudaF_copy_col_from_vec(Gr,Bl,mat,v,col,d); }
template<> inline void cuda_apply_exp<float>(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) { cudaF_apply_exp(Gr,Bl,mat,d); }
template<> inline void cuda_sum<float>(dim3 Gr, dim3 Bl, float* mat, float* value, MatrixDim d) { cudaF_sum(Gr,Bl,mat,value,d); }
template<> inline void cuda_apply_pow<float>(dim3 Gr, dim3 Bl, float* mat, float power, MatrixDim dim) { cudaF_apply_pow(Gr,Bl,mat,power,dim); }
template<> inline void cuda_apply_floor<float>(dim3 Gr, dim3 Bl, float* mat, float floor_val, MatrixDim dim) { cudaF_apply_floor(Gr,Bl,mat,floor_val,dim); }
template<> inline void cuda_trace<float>(int Gr, int Bl, float* mat, float* value, int dim) { cudaF_trace(Gr,Bl,mat,value,dim); }
template<> inline void cuda_set_diag<float>(int Gr, int Bl, float* mat, float value, MatrixDim d) { cudaF_set_diag(Gr,Bl,mat,value,d); }
template<> inline void cuda_set_diag_packed<float>(int Gr, int Bl, float* mat, float value, int dim) { cudaF_set_diag_packed(Gr,Bl,mat,value,dim); }
template<> inline void cuda_add_diag_packed<float>(int Gr, int Bl, float* mat, float value, int dim) { cudaF_add_diag_packed(Gr,Bl,mat,value,dim); }
template<> inline void cuda_set_const<float>(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_set_const(Gr,Bl,mat,value,d); }
template<> inline void cuda_set_zero_above_diag<float>(dim3 Gr, dim3 Bl, float* mat, MatrixDim d) { cudaF_set_zero_above_diag(Gr,Bl,mat,d); }
template<> inline void cuda_add<float>(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_add(Gr,Bl,mat,value,d); }
template<> inline void cuda_add_vec2<float>(dim3 Gr, dim3 Bl, float *mat, const float *vec, const float alpha, int dim) { cudaF_add_vec2(Gr,Bl,mat,vec,alpha,dim); }
template<> inline void cuda_scale_diag<float>(int Gr, int Bl, float* mat, float value, int dim) { cudaF_scale_diag(Gr,Bl,mat,value,dim); }
template<> inline void cuda_scale<float>(dim3 Gr, dim3 Bl, float *mat, float value, MatrixDim d) { cudaF_scale(Gr,Bl,mat,value,d); }
template<> inline void cuda_apply_log<float>(dim3 Gr, dim3 Bl, float *mat, MatrixDim d) { cudaF_apply_log(Gr,Bl,mat,d); }
template<> inline void cuda_mul_elements<float>(dim3 Gr, dim3 Bl, float *mat, const float *A, MatrixDim d) { cudaF_mul_elements(Gr,Bl,mat,A,d); }
template<> inline void cuda_mul_cols_vec<float>(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d) { cudaF_mul_cols_vec(Gr,Bl,mat,scale,d); }
template<> inline void cuda_mul_rows_vec<float>(dim3 Gr, dim3 Bl, float *mat, const float *scale, MatrixDim d) { cudaF_mul_rows_vec(Gr,Bl,mat,scale,d); }
template<> inline void cuda_div_rows_vec<float>(dim3 Gr, dim3 Bl, float *mat, const float *vec_div, MatrixDim d) { cudaF_div_rows_vec(Gr,Bl,mat,vec_div,d); }
template<> inline void cuda_add_mat<float>(dim3 Gr, dim3 Bl, float alpha, const float *A, float beta, float *dst, MatrixDim d) { cudaF_add_mat(Gr,Bl,alpha,A,beta,dst,d); }
template<> inline void cuda_add_vec_to_cols<float>(dim3 Gr, dim3 Bl, float alpha, const float *col, float beta, float *dst, MatrixDim d) { cudaF_add_vec_to_cols(Gr,Bl,alpha,col,beta,dst,d); }
template<> inline void cuda_add_vec_to_rows<float>(dim3 Gr, dim3 Bl, float alpha, const float *row, float beta, float *dst, MatrixDim d) { cudaF_add_vec_to_rows(Gr,Bl,alpha,row,beta,dst,d); }
 
/*
 * CuVector
 */
template<> inline void cuda_set_bias_params<float>(int Gr, int Bl, float* v, const float* a, float param_1, float param_2, float param_3, int* flag, int dim) { cudaF_set_bias_params(Gr,Bl,v,a,param_1,param_2,param_3,flag,dim); }
template<> inline void cuda_copy_from_vec_df<float>(int Gr, int Bl, double* v_out, const float* v_in, int dim) { cudaF_copy_from_vec_df(Gr,Bl,v_out,v_in,dim); }
template<> inline void cuda_copy_from_vec_fd<float>(int Gr, int Bl, float* v_out, const float* v_in, int dim) { cudaF_copy_from_vec_fd(Gr,Bl,v_out,v_in,dim); }
template<> inline void cuda_vec_mul_elements<float>(int Gr, int Bl, float* v, const float* a, int dim) { cudaF_vec_mul_elements(Gr,Bl,v,a,dim); }
template<> inline void cuda_vec_soft_max<float>(int Gr, int Bl, float* v, int dim) { cudaF_vec_soft_max(Gr,Bl,v,dim); }
template<> inline void cuda_min<float>(int Gr, int Bl, const float* v, float* value, int dim) { cudaF_min(Gr,Bl,v,value,dim); }
template<> inline void cuda_trace_mat_mat_trans<float>(int Gr, int Bl, const float* A, const float* B, MatrixDim dA, MatrixDim dB, float* value) { cudaF_trace_mat_mat_trans(Gr,Bl,A,B,dA,dB,value); }
template<> inline void cuda_trace_mat_mat<float>(int Gr, int Bl, const float* A, const float* B, MatrixDim dA, MatrixDim dB, float* value) { cudaF_trace_mat_mat(Gr,Bl,A,B,dA,dB,value); }
template<> inline void cuda_add_diag_mat_trans<float>(int Gr, int Bl, float alpha, float* v, const float* mat, float beta, MatrixDim dmat, int dim) { cudaF_add_diag_mat_trans(Gr,Bl,alpha,v,mat,beta,dmat,dim); }
template<> inline void cuda_add_diag_mat<float>(int Gr, int Bl, float alpha, float* v, const float* mat, float beta, MatrixDim dmat, int dim) { cudaF_add_diag_mat(Gr,Bl,alpha,v,mat,beta,dmat,dim); }
template<> inline void cuda_add_vec_vec<float>(int Gr, int Bl, float alpha, float* v, const float* x, const float* y, float beta, int dim) { cudaF_add_vec_vec(Gr,Bl,alpha,v,x,y,beta,dim); }
template<> inline void cuda_copy_col_from_mat<float>(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim) { cudaF_copy_col_from_mat(Gr,Bl,v,col,mat,dmat,dim); }
template<> inline void cuda_copy_col_from_mat_df<float>(int Gr, int Bl, double* v, int col, const float* mat, MatrixDim dmat, int dim) { cudaF_copy_col_from_mat_df(Gr,Bl,v,col,mat,dmat,dim); }
template<> inline void cuda_copy_col_from_mat_fd<float>(int Gr, int Bl, float* v, int col, const float* mat, MatrixDim dmat, int dim) { cudaF_copy_col_from_mat_fd(Gr,Bl,v,col,mat,dmat,dim); }
template<> inline void cuda_vec_sum<float>(int Gr, int Bl, float* v, float* value, int dim) { cudaF_vec_sum(Gr,Bl,v,value,dim); }
template<> inline void cuda_vec_apply_floor<float>(int Gr, int Bl, float* v, float floor_val, int* num, int dim) { cudaF_vec_apply_floor(Gr,Bl,v,floor_val,num,dim); }
template<> inline void cuda_vec_apply_exp<float>(int Gr, int Bl, float* v, int dim) { cudaF_vec_apply_exp(Gr,Bl,v,dim); }
template<> inline void cuda_vec_apply_log<float>(int Gr, int Bl, float* v, float* flag, int dim) { cudaF_vec_apply_log(Gr,Bl,v,flag,dim); }
template<> inline void cuda_add_row_sum_mat<float>(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d) { cudaF_add_row_sum_mat(Gr,Bl,mat,vec_sum,d); }
template<> inline void cuda_add_col_sum_mat<float>(dim3 Gr, dim3 Bl, const float *mat, float *vec_sum, MatrixDim d) { cudaF_add_col_sum_mat(Gr,Bl,mat,vec_sum,d); }
template<> inline void cuda_invert_elements<float>(dim3 Gr, dim3 Bl, float *data, MatrixDim d) { cudaF_invert_elements(Gr,Bl,data,d); }

/*
 * cu::
 */
template<> inline void cuda_sigmoid<float>(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d) { cudaF_sigmoid(Gr,Bl,y,x,d); }
template<> inline void cuda_diff_sigmoid<float>(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d) { cudaF_diff_sigmoid(Gr,Bl,eout,e,y,d); }
template<> inline void cuda_tanh<float>(dim3 Gr, dim3 Bl, float *y, const float *x, MatrixDim d) { cudaF_tanh(Gr,Bl,y,x,d); }
template<> inline void cuda_diff_tanh<float>(dim3 Gr, dim3 Bl, float *eout, const float *e, const float *y, MatrixDim d) { cudaF_diff_tanh(Gr,Bl,eout,e,y,d); }
template<> inline void cuda_softmax<float>(size_t Gr, size_t Bl, float *y, const float *x, MatrixDim d) { cudaF_softmax(Gr,Bl,y,x,d); }
template<> inline void cuda_softmax_part<float>(dim3 Gr, dim3 Bl, const float *X, const int32_cuda *vec_ids, float* Y, MatrixDim d) { cudaF_softmax_part(Gr,Bl,X,vec_ids,Y,d); }

template<> inline void cuda_regularize_l1<float>(dim3 Gr, dim3 Bl, float *wei, float *grad, float l1, float lr, MatrixDim d) { cudaF_regularize_l1(Gr,Bl,wei,grad,l1,lr,d); }
template<> inline void cuda_find_row_max_id<float>(dim3 Gr, dim3 Bl, const float *mat, float *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { cudaF_find_row_max_id(Gr,Bl,mat,vec_val,vec_id,voff,d); }
template<> inline void cuda_diff_xent<float>(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, float *mat_net_out, float *vec_log_post, MatrixDim d) { cudaF_diff_xent(Gr,Bl,vec_tgt,mat_net_out,vec_log_post,d); }

template<> inline void cuda_randomize<float>(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaF_randomize(Gr,Bl,y,x,copy_from,d_out,d_in); }

template<> inline void cuda_splice<float>(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { cudaF_splice(Gr,Bl,y,x,off,d_out,d_in); }
template<> inline void cuda_one<float>(int Gr,int Bl,float* x,int dim) { cudaF_one(Gr,Bl,x,dim); }
template<> inline void cuda_copy<float>(dim3 Gr, dim3 Bl, float *y, const float *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaF_copy(Gr,Bl,y,x,copy_from,d_out,d_in); }
template<> inline void cuda_copy_diag<float>(int Gr, int Bl, float* y, const float* x, int dim) { cudaF_copy_diag(Gr,Bl,y,x,dim); }
template<> inline void cuda_copy_from_sp<float>(int Gr, int Bl, const float* x, float* y, int d_in, MatrixDim d_out) { cudaF_copy_from_sp(Gr,Bl,x,y,d_in,d_out); }
template<> inline void cuda_take_lower<float>(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out) { cudaF_take_lower(Gr,Bl,x,y,d_in,d_out); }
template<> inline void cuda_take_upper<float>(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out) { cudaF_take_upper(Gr,Bl,x,y,d_in,d_out); }
template<> inline void cuda_take_mean(dim3 Gr, dim3 Bl, const float* x, float* y, MatrixDim d_in, int d_out) { cudaF_take_mean(Gr,Bl,x,y,d_in,d_out); }
/*********************************************************
 * double specializations
 */

/*
 * CuMatrix 
 */
template<> inline void cuda_ammdm_elements<double>(dim3 Gr, dim3 Bl, double alpha, double* mat, const double* A, const double* B, const double* C, double beta, MatrixDim d) { cudaD_ammdm_elements(Gr,Bl,alpha,mat,A,B,C,beta,d); }
template<> inline void cuda_copy_from_tp_trans<double>(int Gr, int Bl, double* A, const double* B, MatrixDim dmat) { cudaD_copy_from_tp_trans(Gr,Bl,A,B,dmat); }
template<> inline void cuda_copy_from_tp<double>(int Gr, int Bl, double* A, const double* B, MatrixDim dmat) { cudaD_copy_from_tp(Gr,Bl,A,B,dmat); }
template<> inline void cuda_trace_sp_sp_fd<double>(int Gr, int Bl, const float* A, const double* B, float* value, int dim) { cudaD_trace_sp_sp_fd(Gr,Bl,A,B,value,dim); }
template<> inline void cuda_trace_sp_sp_df<double>(int Gr, int Bl, const double* A, const double* B, double* value, int dim) { cudaD_trace_sp_sp_df(Gr,Bl,A,B,value,dim); }
template<> inline void cuda_copy_from_mat_fd<double>(dim3 Gr, dim3 Bl, float* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) { cudaD_copy_from_mat_fd(Gr,Bl,mat_out,mat_in,d_out,d_in); }
template<> inline void cuda_copy_from_mat_df<double>(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) { cudaD_copy_from_mat_df(Gr,Bl,mat_out,mat_in,d_out,d_in); }
template<> inline void cuda_copy_from_mat_fd_trans<double>(dim3 Gr, dim3 Bl, float* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) { cudaD_copy_from_mat_fd_trans(Gr,Bl,mat_out,mat_in,d_out,d_in); }
template<> inline void cuda_copy_from_mat_df_trans<double>(dim3 Gr, dim3 Bl, double* mat_out, const double* mat_in, MatrixDim d_out, MatrixDim d_in) { cudaD_copy_from_mat_df_trans(Gr,Bl,mat_out,mat_in,d_out,d_in); }
template<> inline void cuda_copy_col_from_vec<double>(int Gr, int Bl, double* mat, const double* v, int col, MatrixDim d) { cudaD_copy_col_from_vec(Gr,Bl,mat,v,col,d); }
template<> inline void cuda_apply_exp<double>(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) { cudaD_apply_exp(Gr,Bl,mat,d); }
template<> inline void cuda_sum<double>(dim3 Gr, dim3 Bl, double* mat, double* value, MatrixDim d) { cudaD_sum(Gr,Bl,mat,value,d); }
template<> inline void cuda_apply_pow<double>(dim3 Gr, dim3 Bl, double* mat, double power, MatrixDim dim) { cudaD_apply_pow(Gr,Bl,mat,power,dim); }
template<> inline void cuda_apply_floor<double>(dim3 Gr, dim3 Bl, double* mat, double floor_val, MatrixDim dim) { cudaD_apply_floor(Gr,Bl,mat,floor_val,dim); }
template<> inline void cuda_trace<double>(int Gr, int Bl, double* mat, double* value, int dim) { cudaD_trace(Gr,Bl,mat,value,dim); }
template<> inline void cuda_set_diag<double>(int Gr, int Bl, double* mat, double value, MatrixDim d) { cudaD_set_diag(Gr,Bl,mat,value,d); }
template<> inline void cuda_set_diag_packed<double>(int Gr, int Bl, double* mat, double value, int dim) { cudaD_set_diag_packed(Gr,Bl,mat,value,dim); }
template<> inline void cuda_add_diag_packed<double>(int Gr, int Bl, double* mat, double value, int dim) { cudaD_add_diag_packed(Gr,Bl,mat,value,dim); }
template<> inline void cuda_set_const<double>(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_set_const(Gr,Bl,mat,value,d); }
template<> inline void cuda_set_zero_above_diag<double>(dim3 Gr, dim3 Bl, double* mat, MatrixDim d) { cudaD_set_zero_above_diag(Gr,Bl,mat,d); }
template<> inline void cuda_add<double>(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_add(Gr,Bl,mat,value,d); }
template<> inline void cuda_add_vec2<double>(dim3 Gr, dim3 Bl, double *mat, const double *vec, const double alpha, int dim) { cudaD_add_vec2(Gr,Bl,mat,vec,alpha,dim); }
template<> inline void cuda_scale_diag<double>(int Gr, int Bl, double* mat, double value, int dim) { cudaD_scale_diag(Gr,Bl,mat,value,dim); }
template<> inline void cuda_scale<double>(dim3 Gr, dim3 Bl, double *mat, double value, MatrixDim d) { cudaD_scale(Gr,Bl,mat,value,d); }
template<> inline void cuda_apply_log<double>(dim3 Gr, dim3 Bl, double *mat, MatrixDim d) { cudaD_apply_log(Gr,Bl,mat,d); }
template<> inline void cuda_mul_elements<double>(dim3 Gr, dim3 Bl, double *mat, const double *A, MatrixDim d) { cudaD_mul_elements(Gr,Bl,mat,A,d); }
template<> inline void cuda_mul_cols_vec<double>(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d) { cudaD_mul_cols_vec(Gr,Bl,mat,scale,d); }
template<> inline void cuda_mul_rows_vec<double>(dim3 Gr, dim3 Bl, double *mat, const double *scale, MatrixDim d) { cudaD_mul_rows_vec(Gr,Bl,mat,scale,d); }
template<> inline void cuda_div_rows_vec<double>(dim3 Gr, dim3 Bl, double *mat, const double *vec_div, MatrixDim d) { cudaD_div_rows_vec(Gr,Bl,mat,vec_div,d); }
template<> inline void cuda_add_mat<double>(dim3 Gr, dim3 Bl, double alpha, const double *A, double beta, double *dst, MatrixDim d) { cudaD_add_mat(Gr,Bl,alpha,A,beta,dst,d); }
template<> inline void cuda_add_vec_to_cols<double>(dim3 Gr, dim3 Bl, double alpha, const double *col, double beta, double *dst, MatrixDim d) { cudaD_add_vec_to_cols(Gr,Bl,alpha,col,beta,dst,d); }
template<> inline void cuda_add_vec_to_rows<double>(dim3 Gr, dim3 Bl, double alpha, const double *row, double beta, double *dst, MatrixDim d) { cudaD_add_vec_to_rows(Gr,Bl,alpha,row,beta,dst,d); }
 
/*
 * CuVector
 */
template<> inline void cuda_set_bias_params(int Gr, int Bl, double* v, const double* a, double param_1, double param_2, double param_3, int* flag, int dim) { cudaD_set_bias_params(Gr,Bl,v,a,param_1,param_2,param_3,flag,dim); }
template<> inline void cuda_copy_from_vec_df(int Gr, int Bl, double* v_out, const double* v_in, int dim) { cudaD_copy_from_vec_df(Gr,Bl,v_out,v_in,dim); }
template<> inline void cuda_copy_from_vec_fd(int Gr, int Bl, float* v_out, const double* v_in, int dim) { cudaD_copy_from_vec_fd(Gr,Bl,v_out,v_in,dim); }
template<> inline void cuda_vec_mul_elements(int Gr, int Bl, double* v, const double* a, int dim) { cudaD_vec_mul_elements(Gr,Bl,v,a,dim); }
template<> inline void cuda_vec_soft_max(int Gr, int Bl, double* v, int dim) { cudaD_vec_soft_max(Gr,Bl,v,dim); }
template<> inline void cuda_min<double>(int Gr, int Bl, const double* v, double* value, int dim) { cudaD_min(Gr,Bl,v,value,dim); }
template<> inline void cuda_trace_mat_mat_trans<double>(int Gr, int Bl, const double* A, const double* B, MatrixDim dA, MatrixDim dB, double* value) { cudaD_trace_mat_mat_trans(Gr,Bl,A,B,dA,dB,value); }
template<> inline void cuda_trace_mat_mat<double>(int Gr, int Bl, const double* A, const double* B, MatrixDim dA, MatrixDim dB, double* value) { cudaD_trace_mat_mat(Gr,Bl,A,B,dA,dB,value); }
template<> inline void cuda_add_diag_mat_trans<double>(int Gr, int Bl, double alpha, double* v, const double* mat, double beta, MatrixDim dmat, int dim) { cudaD_add_diag_mat_trans(Gr,Bl,alpha,v,mat,beta,dmat,dim); }
template<> inline void cuda_add_diag_mat<double>(int Gr, int Bl, double alpha, double* v, const double* mat, double beta, MatrixDim dmat, int dim) { cudaD_add_diag_mat(Gr,Bl,alpha,v,mat,beta,dmat,dim); }
template<> inline void cuda_add_vec_vec<double>(int Gr, int Bl, double alpha, double* v, const double* x, const double* y, double beta, int dim) { cudaD_add_vec_vec(Gr,Bl,alpha,v,x,y,beta,dim); }
template<> inline void cuda_copy_col_from_mat<double>(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim) { cudaD_copy_col_from_mat(Gr,Bl,v,col,mat,dmat,dim); }
template<> inline void cuda_copy_col_from_mat_df<double>(int Gr, int Bl, double* v, int col, const double* mat, MatrixDim dmat, int dim) { cudaD_copy_col_from_mat_df(Gr,Bl,v,col,mat,dmat,dim); }
template<> inline void cuda_copy_col_from_mat_fd<double>(int Gr, int Bl, float* v, int col, const double* mat, MatrixDim dmat, int dim) { cudaD_copy_col_from_mat_fd(Gr,Bl,v,col,mat,dmat,dim); }
template<> inline void cuda_vec_sum<double>(int Gr, int Bl, double* v, double* value, int dim) { cudaD_vec_sum(Gr,Bl,v,value,dim); }
template<> inline void cuda_vec_apply_floor<double>(int Gr, int Bl, double* v, double floor_val, int* num, int dim) { cudaD_vec_apply_floor(Gr,Bl,v,floor_val,num,dim); }
template<> inline void cuda_vec_apply_exp<double>(int Gr, int Bl, double* v, int dim) { cudaD_vec_apply_exp(Gr,Bl,v,dim); }
template<> inline void cuda_vec_apply_log<double>(int Gr, int Bl, double* v, double* flag, int dim) { cudaD_vec_apply_log(Gr,Bl,v,flag,dim); }
template<> inline void cuda_add_row_sum_mat<double>(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d) { cudaD_add_row_sum_mat(Gr,Bl,mat,vec_sum,d); }
template<> inline void cuda_add_col_sum_mat<double>(dim3 Gr, dim3 Bl, const double *mat, double *vec_sum, MatrixDim d) { cudaD_add_col_sum_mat(Gr,Bl,mat,vec_sum,d); }
template<> inline void cuda_invert_elements<double>(dim3 Gr, dim3 Bl, double *data, MatrixDim d) { cudaD_invert_elements(Gr,Bl,data,d); }

/*
 * cu::
 */
template<> inline void cuda_sigmoid<double>(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d) { cudaD_sigmoid(Gr,Bl,y,x,d); }
template<> inline void cuda_diff_sigmoid<double>(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d) { cudaD_diff_sigmoid(Gr,Bl,eout,e,y,d); }
template<> inline void cuda_tanh<double>(dim3 Gr, dim3 Bl, double *y, const double *x, MatrixDim d) { cudaD_tanh(Gr,Bl,y,x,d); }
template<> inline void cuda_diff_tanh<double>(dim3 Gr, dim3 Bl, double *eout, const double *e, const double *y, MatrixDim d) { cudaD_diff_tanh(Gr,Bl,eout,e,y,d); }
template<> inline void cuda_softmax<double>(size_t Gr, size_t Bl, double *y, const double *x, MatrixDim d) { cudaD_softmax(Gr,Bl,y,x,d); }
template<> inline void cuda_softmax_part<double>(dim3 Gr, dim3 Bl, const double *X, const int32_cuda *vec_ids, double* Y, MatrixDim d) { cudaD_softmax_part(Gr,Bl,X,vec_ids,Y,d); }

template<> inline void cuda_regularize_l1<double>(dim3 Gr, dim3 Bl, double *wei, double *grad, double l1, double lr, MatrixDim d) { cudaD_regularize_l1(Gr,Bl,wei,grad,l1,lr,d); }
template<> inline void cuda_find_row_max_id<double>(dim3 Gr, dim3 Bl, const double *mat, double *vec_val, int32_cuda *vec_id, int32_cuda voff, MatrixDim d) { cudaD_find_row_max_id(Gr,Bl,mat,vec_val,vec_id,voff,d); }
template<> inline void cuda_diff_xent<double>(dim3 Gr, dim3 Bl, const int32_cuda *vec_tgt, double *mat_net_out, double *vec_log_post, MatrixDim d) { cudaD_diff_xent(Gr,Bl,vec_tgt,mat_net_out,vec_log_post,d); }

template<> inline void cuda_randomize<double>(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaD_randomize(Gr,Bl,y,x,copy_from,d_out,d_in); }
template<> inline void cuda_splice<double>(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *off, MatrixDim d_out, MatrixDim d_in) { cudaD_splice(Gr,Bl,y,x,off,d_out,d_in); }
template<> inline void cuda_one<double>(int Gr,int Bl,double* x,int dim) { cudaD_one(Gr,Bl,x,dim); }
template<> inline void cuda_copy<double>(dim3 Gr, dim3 Bl, double *y, const double *x, const int32_cuda *copy_from, MatrixDim d_out, MatrixDim d_in) { cudaD_copy(Gr,Bl,y,x,copy_from,d_out,d_in); }
template<> inline void cuda_copy_diag<double>(int Gr, int Bl, double* y, const double* x, int dim) { cudaD_copy_diag(Gr,Bl,y,x,dim); }
template<> inline void cuda_copy_from_sp<double>(int Gr, int Bl, const double* x, double* y, int d_in, MatrixDim d_out) { cudaD_copy_from_sp(Gr,Bl,x,y,d_in,d_out); }
template<> inline void cuda_take_lower<double>(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out) { cudaD_take_lower(Gr,Bl,x,y,d_in,d_out); }
template<> inline void cuda_take_upper<double>(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out) { cudaD_take_upper(Gr,Bl,x,y,d_in,d_out); }
template<> inline void cuda_take_mean<double>(dim3 Gr, dim3 Bl, const double* x, double* y, MatrixDim d_in, int d_out) { cudaD_take_mean(Gr,Bl,x,y,d_in,d_out); }
} // namespace



#endif // HAVE_CUDA

#endif

