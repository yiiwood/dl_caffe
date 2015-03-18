#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ResidualForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* bottom_label, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int predt = static_cast<int>(bottom_data[index]);
    int label = static_cast<int>(bottom_label[index]);
    top_data[index] = label - predt;
  }
}

template <typename Dtype>
void ResidualValueLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();

  int count = bottom[0]->count();
  ResidualForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom_label, top_data);
}

template <typename Dtype>
void ResidualValueLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  // TODO(Yangqing): implement the GPU version of softmax.
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(ResidualValueLayer);

}  // namespace caffe
