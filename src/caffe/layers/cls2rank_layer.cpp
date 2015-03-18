#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Cls2RankLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Cls2Rank Layer takes one blobs as input.";
  CHECK_EQ(top->size(), 1) << "Cls2Rank Layer takes one output.";
}

template <typename Dtype>
void Cls2RankLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void Cls2RankLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();

  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < spatial_dim; k++) {
      top_data[i * dim + (channels - 1) * spatial_dim + k] = 1.0 - bottom_data[i * dim + (channels - 1) * spatial_dim + k];
      for (int j = 0; j < (channels - 1); ++j) {
        top_data[i * dim + (j + 1) * spatial_dim + k] = 1.0 - bottom_data[i * dim + j * spatial_dim + k];
      }
    }
  }
}

INSTANTIATE_CLASS(Cls2RankLayer);

}  // namespace caffe
