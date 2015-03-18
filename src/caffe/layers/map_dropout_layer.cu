#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void MapDropoutForward(const int nc, const int spatial_dim, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, nc) {
    const Dtype* in_s = in + index * spatial_dim;
    Dtype* out_s = out + index * spatial_dim;
    if (mask[index] > threshold) {
      for (int i = 0; i < spatial_dim; ++i) {
        out_s[i] = in_s[i] * scale;
      }
    } else {
      for (int i = 0; i < spatial_dim; ++i) { 
        out_s[i] = Dtype(0.);
      }
    }
  }
}

template <typename Dtype>
void MapDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int nc = num * channels;
  const int spatial_dim = bottom[0]->count() / nc;
  if (Caffe::phase() == Caffe::TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(nc, mask);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    MapDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
        nc, spatial_dim, bottom_data, mask, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void MapDropoutBackward(const int nc, const int spatial_dim, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, nc) {
    const Dtype* in_diff_s = in_diff + index * spatial_dim;
    Dtype* out_diff_s = out_diff + index * spatial_dim;
    if (mask[index] > threshold) {
      for (int i = 0; i < spatial_dim; ++i) {
        out_diff_s[i] = in_diff_s[i] * scale;
      }
    } else {
      for (int i = 0; i < spatial_dim; ++i) {
        out_diff_s[i] = Dtype(0.);
      }
    }
  }
}

template <typename Dtype>
void MapDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    if (Caffe::phase() == Caffe::TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int num = (*bottom)[0]->num();
      const int channels = (*bottom)[0]->channels();
      const int nc = num * channels;
      const int spatial_dim = (*bottom)[0]->count() / nc;
      // NOLINT_NEXT_LINE(whitespace/operators)
      MapDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(nc), CAFFE_CUDA_NUM_THREADS>>>(
          nc, spatial_dim, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(MapDropoutLayer);


}  // namespace caffe
