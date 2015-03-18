// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MapDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void MapDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void MapDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int nc = num * channels;
  const int spatial_dim = bottom[0]->count() / nc;
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(nc, 1. - threshold_, mask);
    for (int i = 0; i < nc; ++i) {
      if ( mask[i] ) {
        caffe_copy(spatial_dim, bottom_data + i * spatial_dim, top_data + i * spatial_dim);
        caffe_scal(spatial_dim, scale_, top_data + i * spatial_dim);
      } else {
        caffe_set(spatial_dim, Dtype(0.), top_data + i * spatial_dim);
      }
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void MapDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    if (Caffe::phase() == Caffe::TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int num = (*bottom)[0]->num();
      const int channels = (*bottom)[0]->channels();
      const int nc = num * channels;
      const int spatial_dim = (*bottom)[0]->count() / nc;
      for (int i = 0; i < nc; ++i) {
        if ( mask[i] ) {
          caffe_copy(spatial_dim, top_diff + i * spatial_dim, bottom_diff + i * spatial_dim);
          caffe_scal(spatial_dim, scale_, bottom_diff + i * spatial_dim);
        } else {
          caffe_set(spatial_dim, Dtype(0.), bottom_diff + i * spatial_dim);
        }
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(MapDropoutLayer);
#endif

INSTANTIATE_CLASS(MapDropoutLayer);


}  // namespace caffe
