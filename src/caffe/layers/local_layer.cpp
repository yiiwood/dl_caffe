// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Configure the kernel size, padding, stride, and inputs.
  LocalParameter local_param = this->layer_param_.local_param();
  CHECK(!local_param.has_kernel_size() !=
      !(local_param.has_kernel_h() && local_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(local_param.has_kernel_size() ||
      (local_param.has_kernel_h() && local_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!local_param.has_pad() && local_param.has_pad_h()
      && local_param.has_pad_w())
      || (!local_param.has_pad_h() && !local_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!local_param.has_stride() && local_param.has_stride_h()
      && local_param.has_stride_w())
      || (!local_param.has_stride_h() && !local_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (local_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = local_param.kernel_size();
  } else {
    kernel_h_ = local_param.kernel_h();
    kernel_w_ = local_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!local_param.has_pad_h()) {
    pad_h_ = pad_w_ = local_param.pad();
  } else {
    pad_h_ = local_param.pad_h();
    pad_w_ = local_param.pad_w();
  }
  if (!local_param.has_stride_h()) {
    stride_h_ = stride_w_ = local_param.stride();
  } else {
    stride_h_ = local_param.stride_h();
    stride_w_ = local_param.stride_w();
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_output_ = this->layer_param_.local_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.local_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  height_out_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  M_ = num_output_ / group_;
  K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  N_ = height_out_ * width_out_;
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.local_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    this->blobs_[0].reset(new Blob<Dtype>(
        num_output_, channels_ / group_, kernel_h_ * kernel_w_, N_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.local_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // If necessary, initialize and fill the biases:
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, num_output_, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.local_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " local kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  for (int top_id = 0; top_id < top->size(); ++top_id) {
    (*top)[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  // Prepare the matrix multiplication computation.
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage.
  col_buffer_.Reshape(1, channels_, kernel_h_ * kernel_w_, N_);
  col_tmp_.Reshape(1, channels_, kernel_h_ * kernel_w_, N_);

  col_multiplier_.Reshape(1, 1, 1, K_);
  caffe_set(K_, Dtype(1), col_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void LocalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    Dtype* col_tmp_data = col_tmp_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    int weight_offset_group = M_ * K_ * N_;
    int weight_offset_output = K_ * N_;
    int col_offset = K_ * N_;
    int top_offset_group = M_ * N_;
    int top_offset_output = N_;
    for (int n = 0; n < num_; ++n) {
      // First, im2col
      im2col_cpu(bottom_data + bottom[i]->offset(n), channels_, height_, width_, 
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_data);
      // Second, elementwise product and sum every K_ elements
      for (int g = 0; g < group_; ++g) {
        for (int o = 0; o < M_ ; ++o) {
          caffe_mul<Dtype>(K_ * N_,
            weight + weight_offset_group * g + weight_offset_output * o, 
            col_data + col_offset * g,
            col_tmp_data + col_offset * g);
          caffe_cpu_gemv(CblasTrans, K_, N_,
            (Dtype)1., col_tmp_data + col_offset * g, col_multiplier_.cpu_data(), (Dtype)0.,
            top_data + (*top)[i]->offset(n) + top_offset_group * g + top_offset_output * o);
        }
      }
      // third, add bias
      if (bias_term_) {
        caffe_add<Dtype>(num_output_ * N_, this->blobs_[1]->cpu_data(),
          top_data + (*top)[i]->offset(n), top_data + (*top)[i]->offset(n));
      }
    }
  }
}

template <typename Dtype>
void LocalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
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
      top_diff = top[i]->cpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_add<Dtype>(num_output_ * N_, top_diff + top[i]->offset(n),
          bias_diff, bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->cpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_cpu_data();
      Dtype* col_diff = col_buffer_.mutable_cpu_diff();
      Dtype* col_tmp_diff = col_tmp_.mutable_cpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->cpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_cpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_, width_, 
          kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            for (int o = 0; o < M_ ; ++o) {
              caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K_, N_, 1,
                (Dtype)1., col_multiplier_.cpu_data(), 
                top_diff + top[i]->offset(n) + top_offset_group * g + top_offset_output * o,
                (Dtype)0., col_tmp_diff + col_offset * g);
              caffe_mul<Dtype>(K_ * N_,
                col_tmp_diff + col_offset * g,
                col_data + col_offset * g,
                col_tmp_diff + col_offset * g);
              caffe_add<Dtype>(K_ * N_,
                col_tmp_diff + col_offset * g,
                weight_diff + weight_offset_group * g + weight_offset_output * o,
                weight_diff + weight_offset_group * g + weight_offset_output * o);
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->cpu_data();
          }
          caffe_set(col_buffer_.count(), Dtype(0), col_diff);
          for (int g = 0; g < group_; ++g) {
            for (int o = 0; o < M_ ; ++o) {
              caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K_, N_, 1,
                (Dtype)1., col_multiplier_.cpu_data(), 
                top_diff + top[i]->offset(n) + top_offset_group * g + top_offset_output * o,
                (Dtype)0., col_tmp_diff + col_offset * g);
              caffe_mul<Dtype>(K_ * N_,
                col_tmp_diff + col_offset * g,
                weight + weight_offset_group * g + weight_offset_output * o,
                col_tmp_diff + col_offset * g);
              caffe_add<Dtype>(K_ * N_,
                col_tmp_diff + col_offset * g,
                col_diff + col_offset * g,
                col_diff + col_offset * g);
            }
          }
          // col2im back to the data
          col2im_cpu(col_diff, channels_, height_, width_,
              kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
              bottom_diff + (*bottom)[i]->offset(n));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LocalLayer);
#endif

INSTANTIATE_CLASS(LocalLayer);

}  // namespace caffe
