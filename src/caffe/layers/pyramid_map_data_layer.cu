#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void PyramidMapDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  for (int i = 1; i <= level_num_; ++i) {
    caffe_copy(prefetch_pyramid_data_[i-1]->count(), prefetch_pyramid_data_[i-1]->cpu_data(),
        (*top)[i]->mutable_gpu_data());
  }
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        (*top)[level_num_ + 1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(PyramidMapDataLayer);

}  // namespace caffe
