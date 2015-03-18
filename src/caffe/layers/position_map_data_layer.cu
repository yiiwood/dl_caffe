#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void PositionMapDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  this->JoinPrefetchThread();
  int num = this->prefetch_data_.num();
  int dim = this->prefetch_data_.count() / num; 
  int pdim = 2 * this->prefetch_data_.height() * this->prefetch_data_.width();
  for(int i = 0; i < num; ++i) {
     caffe_copy(dim, this->prefetch_data_.cpu_data() + i * dim,(*top)[0]->mutable_cpu_data() + i * (dim + pdim));
     caffe_copy(pdim, pos_blob->cpu_data(),(*top)[0]->mutable_cpu_data() + i * (dim + pdim) + dim);
  }
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(PositionMapDataLayer);

}  // namespace caffe
