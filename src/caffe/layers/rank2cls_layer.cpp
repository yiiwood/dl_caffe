#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Rank2ClsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Rank2Cls Layer takes one blobs as input.";
  CHECK_EQ(bottom[0]->channels(), 1) << "Rank2Cls Layer takes one channel as input.";
  CHECK_EQ(top->size(), 1) << "Rank2Cls Layer takes one output.";
}

template <typename Dtype>
void Rank2ClsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  rank_num_ = this->layer_param_.rank2cls_param().rank_num();
  (*top)[0]->Reshape(bottom[0]->num(), rank_num_, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void Rank2ClsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  int num = bottom[0]->num(); 
  int spatial_dim = bottom[0]->height() * bottom[0]->width();

  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < spatial_dim; k++) {
      int rank = bottom_data[i * spatial_dim + k];
      for ( int r = 0; r < rank_num_; ++r) {
        if (r > rank ) { 
          top_data[i * rank_num_ * spatial_dim + r * spatial_dim + k] = 1;
        }   
        else {
          top_data[i * rank_num_ * spatial_dim + r * spatial_dim + k] = 0;
        }   
      }
    }
  }
}

INSTANTIATE_CLASS(Rank2ClsLayer);

}  // namespace caffe
