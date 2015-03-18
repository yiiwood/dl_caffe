#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AverageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_GE(bottom.size(), 2);               // now support 2
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[0]->num(), bottom[i]->num());
    CHECK_EQ(bottom[0]->channels(), bottom[i]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[i]->height());
    CHECK_EQ(bottom[0]->width(), bottom[i]->width());
  }
}

template <typename Dtype>
void AverageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void AverageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int count = bottom[0]->count();
  int size = bottom.size();
  caffe_add(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
  for (int i = 2; i < size; ++i) {
    caffe_add(count, bottom[i]->cpu_data(), top_data, top_data);
  }
  caffe_scal(count, Dtype(1.0) / size, top_data);
}

template <typename Dtype>
void AverageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  int count = (*bottom)[0]->count();
  int size = bottom->size();
  for (int i = 0; i < size; ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
      caffe_copy(count, top_diff, bottom_diff);
      caffe_scal(count, Dtype(1.0) / size, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AverageLayer);
#endif

INSTANTIATE_CLASS(AverageLayer);

}  // namespace caffe
