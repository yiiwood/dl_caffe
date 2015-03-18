#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  int channels = prob_.channels();
  int accuracy = 0;
  int valid_num = 0;

  vector<Dtype> label_num(channels, 0); 
  vector<Dtype> accuracy_num(channels, 0); 

  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      // --------------------- mask -----------------------//
      int lb = static_cast<int>(label[i * spatial_dim + j]);
      if ( lb < 0 || lb >= channels ) continue;
      // --------------------- mask -----------------------//

      loss -= log(std::max(prob_data[i * dim +
          static_cast<int>(label[i * spatial_dim + j]) * spatial_dim + j],
                           Dtype(FLT_MIN)));

      Dtype maxval = -FLT_MAX;
      int max_id = -1;
      for (int c = 0; c < channels; ++c) {
        int idx = i * dim + c * spatial_dim + j;
        if ( prob_data[idx] > maxval) {
          maxval = prob_data[idx];
          max_id = c;
        }
      }
      if (max_id == lb) {
        ++accuracy;
        ++accuracy_num[ max_id ];
      }
      ++valid_num;
      ++label_num[ lb ];

    }
  }

  (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
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
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    //LOG(FATAL) << this->type_name()
    //           << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    int channels = prob_.channels();

    vector<Dtype> label_num(channels, 0);
    vector<Dtype> sampling_num(channels, 0);
    vector<Dtype> sampling_prob(channels, 1);
    int max_num = 0;
    int valid_num = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        // --------------------- mask -----------------------//
        int lb = static_cast<int>(label[i * spatial_dim + j]);
        if ( lb < 0 || lb >= channels ) {
           for (int c = 0; c < channels; ++c) {
              bottom_diff[i * dim + c * spatial_dim + j] = 0;
           }
           continue;
        }
        // --------------------- mask -----------------------//
        ++label_num[lb];
        // -------------------- balance ---------------------//
        /*float rd = ((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1));
        if ( rd > sampling_prob[lb]) {
           for (int c = 0; c < channels; ++c) {
              bottom_diff[i * dim + c * spatial_dim + j] = 0;
           }
           continue;
        }*/
        // -------------------- balance ---------------------//
        
        bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
            * spatial_dim + j] -= 1;
        
        ++valid_num;
        ++sampling_num[lb];
        if (sampling_num[lb] > max_num) {
           max_num = sampling_num[lb];
           for (int c = 0; c < channels; ++c) {
              sampling_prob[c] = 1.0 - sampling_num[c] / (max_num + 100.0);
           }   
        }   

      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    //caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
    caffe_scal(prob_.count(), loss_weight / valid_num, bottom_diff);
    
    /*
    for (int c = 0; c < channels; ++c) {
       LOG(INFO) << "grad_num " << c << ": " << sampling_num[c] << " / " << label_num[c];
    }*/

  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
