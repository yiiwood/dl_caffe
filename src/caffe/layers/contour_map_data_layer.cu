#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void ContourMapDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());

  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
    caffe_copy(prefetch_contour_data_.count(), this->prefetch_contour_data_.cpu_data(),
        (*top)[2]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(ContourMapDataLayer);

}  // namespace caffe
