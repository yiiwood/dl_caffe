// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LocalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    Dtype* col_tmp_data = col_tmp_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weight_offset_group = M_ * K_ * N_;
    int weight_offset_output = K_ * N_;
    int col_offset = K_ * N_;
    int top_offset_group = M_ * N_;
    int top_offset_output = N_;
    for (int n = 0; n < num_; ++n) {
      // First, im2col
      im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_, width_, 
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_data);
      // Second, elementwise product and sum every K_ elements
      for (int g = 0; g < group_; ++g) {
        for (int o = 0; o < M_ ; ++o) {
          caffe_gpu_mul<Dtype>(K_ * N_,
            weight + weight_offset_group * g + weight_offset_output * o, 
            col_data + col_offset * g,
            col_tmp_data + col_offset * g);
          caffe_gpu_gemv(CblasTrans, K_, N_,
            (Dtype)1., col_tmp_data + col_offset * g, col_multiplier_.gpu_data(), (Dtype)0.,
            top_data + (*top)[i]->offset(n) + top_offset_group * g + top_offset_output * o);
        }
      }
      // third, add bias
      if (bias_term_) {
        caffe_gpu_add<Dtype>(num_output_ * N_, this->blobs_[1]->gpu_data(),
          top_data + (*top)[i]->offset(n), top_data + (*top)[i]->offset(n));
      }
    }
  }
}

template <typename Dtype>
void LocalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  int weight_offset_group = M_ * K_ * N_;
  int weight_offset_output = K_ * N_;
  int col_offset = K_ * N_;
  int top_offset_group = M_ * N_;
  int top_offset_output = N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1] ) {
      top_diff = top[i]->gpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_add<Dtype>(num_output_ * N_, top_diff + top[i]->offset(n),
          bias_diff, bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->gpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_gpu_data();
      Dtype* col_diff = col_buffer_.mutable_gpu_diff();
      Dtype* col_tmp_diff = col_tmp_.mutable_gpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_, width_, 
          kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            for (int o = 0; o < M_ ; ++o) {
              caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, K_, N_, 1,
                (Dtype)1., col_multiplier_.gpu_data(), 
                top_diff + top[i]->offset(n) + top_offset_group * g + top_offset_output * o,
                (Dtype)0., col_tmp_diff + col_offset * g);
              caffe_gpu_mul<Dtype>(K_ * N_,
                col_tmp_diff + col_offset * g,
                col_data + col_offset * g,
                col_tmp_diff + col_offset * g);
              caffe_gpu_add<Dtype>(K_ * N_,
                col_tmp_diff + col_offset * g,
                weight_diff + weight_offset_group * g + weight_offset_output * o,
                weight_diff + weight_offset_group * g + weight_offset_output * o);
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
          caffe_gpu_set(col_buffer_.count(), Dtype(0), col_diff);
          for (int g = 0; g < group_; ++g) {
            for (int o = 0; o < M_ ; ++o) {
              caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, K_, N_, 1,
                (Dtype)1., col_multiplier_.gpu_data(), 
                top_diff + top[i]->offset(n) + top_offset_group * g + top_offset_output * o,
                (Dtype)0., col_tmp_diff + col_offset * g);
              caffe_gpu_mul<Dtype>(K_ * N_,
                col_tmp_diff + col_offset * g,
                weight + weight_offset_group * g + weight_offset_output * o,
                col_tmp_diff + col_offset * g);
              caffe_gpu_add<Dtype>(K_ * N_,
                col_tmp_diff + col_offset * g,
                col_diff + col_offset * g,
                col_diff + col_offset * g);
            }
          }
          // col2im back to the data
          col2im_gpu(col_diff, channels_, height_, width_,
              kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
              bottom_diff + (*bottom)[i]->offset(n));
        }
      }
    }
  }
}


INSTANTIATE_CLASS(LocalLayer);

}  // namespace caffe
