#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HardNegativeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
}

template <typename Dtype>
void HardNegativeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
  hard_ratio_ = (float)this->layer_param_.hard_negative_param().hard_ratio();
  if (this->layer_param_.hard_negative_param().has_base_num()) {
    base_num_ = this->layer_param_.hard_negative_param().base_num();
  }
  else {
    base_num_ = bottom[1]->count() / (2*bottom[0]->channels());
  }
}

template <typename Dtype>
void HardNegativeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_prob = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_dim = height * width;
  int dim = channels * spatial_dim;

  vector<Dtype> sampling_prob(channels, 1); 
  vector<int> sampling_num(channels, 0);
  int max_num = 0;
  std::vector<std::vector<std::pair<Dtype, int> > > cls_prob(channels);
  for (int n = 0; n < num; ++n) {
    for (int s = 0; s < spatial_dim; ++s) {
      int idx = n * spatial_dim + s;
      int lb = static_cast<int>(bottom_label[idx]);
      if ( lb < 0 || lb >= channels ) continue;
      
      cls_prob[lb].push_back( std::make_pair(bottom_prob[n * dim + lb * spatial_dim + s], idx) );

      // ------------------ sampling mode 0 ------------------ //
      /*float rd = ((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1));
      if (rd < sampling_prob[lb]) sampling_num[lb]++;
      
      if (sampling_num[lb] > 2 * base_num_) {
        sampling_prob[lb] = 0.0;
      }
      else if (cls_prob[lb].size() > base_num_){
        sampling_prob[lb] = base_num_ / (cls_prob[lb].size() * 2.0);
      }*/
      /*
      if (cls_prob[lb].size() > max_num) {
        max_num = cls_prob[lb].size();
        for (int c = 0; c < channels; ++c) {
          if (cls_prob[c].size() <= base_num_) continue;
          sampling_prob[c] = 1.0 - cls_prob[c].size() / Dtype(max_num + 100);
        }
      }*/
      // ------------------ sampling mode 0 ------------------ //

    }
  }

  int min_num = bottom[0]->count();
  for (int i = 0; i < channels; ++i) {
    if (min_num > cls_prob[i].size() ) {
      min_num = cls_prob[i].size();
    }  
  }
  min_num = (min_num <= 5)? 5 : min_num;
  min_num += 1;

  caffe_set((*top)[0]->count(), Dtype(-1), top_data);
  for (int i = 0; i < channels; ++i) {
    // ------------------ sampling mode 1 ------------------ //
    //sampling_num[i] = cls_prob[i].size() > base_num_? base_num_ : cls_prob[i].size();
    // ------------------ sampling mode 1 ------------------ //
    // ------------------ sampling mode 2 ------------------ //
    sampling_num[i] = cls_prob[i].size() > min_num? min_num : cls_prob[i].size();
    // ------------------ sampling mode 2 ------------------ //

    int sample_num = sampling_num[i];
    //int sample_num = cls_prob[i].size();
    std::partial_sort(cls_prob[i].begin(), cls_prob[i].begin() + sample_num, cls_prob[i].end() );

    int hard_num = int(sample_num * hard_ratio_);
    int sample_range = sample_num;
    for (int j = 0; j < hard_num; ++j) {
      //LOG(INFO) << j << ": " << cls_prob[i][j].first;
      int sidx = j + (caffe::caffe_rng_rand() % sample_range);
      int idx = cls_prob[i][sidx].second;
      int s = idx % spatial_dim;
      int n = idx / spatial_dim;
      top_data[n * spatial_dim + s] = i;
      std::swap(cls_prob[i][sidx], cls_prob[i][j]);
      sample_range--;
    }
    int smp_num = sample_num - hard_num;
    int smp_range = cls_prob[i].size() - hard_num;
    for (int j = 0; j < smp_num; ++j) {
      int sidx = hard_num + j + (caffe::caffe_rng_rand() % smp_range);
      int idx = cls_prob[i][sidx].second;
      int s = idx % spatial_dim;
      int n = idx / spatial_dim;
      top_data[n * spatial_dim + s] = i;
      std::swap(cls_prob[i][sidx], cls_prob[i][hard_num + j]);
      smp_range--;
    }
    //LOG(INFO) << sample_num << " = " << hard_num << " + " << smp_num;
  } 

  //for (int i = 0; i < channels; ++i)  LOG(INFO) << i << ": " << sampling_num[i] << " / " << cls_prob[i].size();
}

template <typename Dtype>
void HardNegativeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    // not contribute to gradient!
    caffe_set((*bottom)[0]->count(), Dtype(0.), bottom_diff);  
  }
}

#ifdef CPU_ONLY
STUB_GPU(HardNegativeLayer);
#endif

INSTANTIATE_CLASS(HardNegativeLayer);

}  // namespace caffe
