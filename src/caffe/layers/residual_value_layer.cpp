#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ResidualValueLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
}

template <typename Dtype>
void ResidualValueLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ResidualValueLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int count = bottom[0]->count();
  for (int idx = 0; idx < count; ++idx) {
    Dtype predt = bottom_data[idx];
    Dtype label = bottom_label[idx];
    top_data[idx] = label - predt;
  }
}

template <typename Dtype>
void ResidualValueLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    // not contribute to gradient!
    caffe_set((*bottom)[0]->count(), Dtype(0.), bottom_diff);  
  }
  if (propagate_down[1]) {
    Dtype* bottom_diff = (*bottom)[1]->mutable_cpu_diff();
    // not contribute to gradient!
    caffe_set((*bottom)[1]->count(), Dtype(0.), bottom_diff);  
  }
}

#ifdef CPU_ONLY
STUB_GPU(ResidualValueLayer);
#endif

INSTANTIATE_CLASS(ResidualValueLayer);

}  // namespace caffe
