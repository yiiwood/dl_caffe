#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MapHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();

  int accuracy = 0;
  int valid_num = 0;
  vector<Dtype> label_num(channels, 0);  
  vector<Dtype> accuracy_num(channels, 0);  

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      int lb = static_cast<int>(label[i * spatial_dim + j]);
      // --------------------- mask -----------------------//
      if ( lb < 0 || lb >= channels ) {
        for (int c = 0; c < channels; ++c) {
          bottom_diff[i * dim + c * spatial_dim + j] = 0;
        }
        continue;
      }
      // --------------------- mask -----------------------//

      ++valid_num;
      int idx = i * dim + lb * spatial_dim + j;
      bottom_diff[idx] *= -1;

      Dtype maxval = -FLT_MAX;
      int max_id = -1;
      for (int c = 0; c < channels; ++c) {
        int idx = i * dim + c * spatial_dim + j; 
        if ( bottom_data[idx] > maxval) {
          maxval = bottom_data[idx];
          max_id = c; 
        }    
      }    
      if (max_id == lb) {
        ++accuracy;
        ++accuracy_num[ max_id ];
      }    
      ++label_num[ lb ];
    }
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = std::max(
        Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }

  Dtype* loss = (*top)[0]->mutable_cpu_data();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, bottom_diff) / valid_num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / valid_num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }

  Dtype sum = 0.;
  for (int c = 0; c < channels; ++c) {
    LOG(INFO) << "accuracy_num " << c << ": " << accuracy_num[c] << " / " << label_num[c];                                   
    if ( label_num[c] > 0) {
      sum += accuracy_num[c] / label_num[c];
    }    
  }      
  LOG(INFO) << "global_accuracy = " << accuracy*1.0 / valid_num <<", average_accuracy = " << sum / channels;
}

template <typename Dtype>
void MapHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = (*bottom)[0]->num();
    int count = (*bottom)[0]->count();
    int dim = count / num;
    int spatial_dim = (*bottom)[0]->height() * (*bottom)[0]->width();
    int channels = (*bottom)[0]->channels();

    int valid_num = 0;

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
        int lb = static_cast<int>(label[i * spatial_dim + j]);

        if ( lb < 0 || lb >= channels ) {
          for (int c = 0; c < channels; ++c) {
            bottom_diff[i * dim + c * spatial_dim + j] = 0;
          }
          continue;
        }

        ++valid_num;
        int idx = i * dim + lb * spatial_dim + j;
        bottom_diff[idx] *= -1;
      }
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / valid_num, bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / valid_num, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(MapHingeLossLayer);

}  // namespace caffe
