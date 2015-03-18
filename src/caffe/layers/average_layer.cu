#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AverageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = bottom[0]->count();
  int size = bottom.size();
  caffe_gpu_add(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
  for (int i = 2; i < size; ++i) {
    caffe_gpu_add(count, bottom[i]->gpu_data(), top_data, top_data);
  }
  caffe_gpu_scal(count, Dtype(1.0) / size, top_data);
}

template <typename Dtype>
void AverageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  int count = (*bottom)[0]->count();
  int size = bottom->size();
  for (int i = 0; i < size; ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      caffe_copy(count, top_diff, bottom_diff);
      caffe_gpu_scal(count, Dtype(1.0) / size, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(AverageLayer);

}  // namespace caffe
