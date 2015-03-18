#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MapAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  top_k_ = this->layer_param_.map_accuracy_param().top_k();
  global_ = this->layer_param_.map_accuracy_param().global();
  average_ = this->layer_param_.map_accuracy_param().average();

  CHECK_EQ(bottom.size(), 2) << "MapAccuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "MapAccuracy Layer takes one output.";

  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_GE(top_k_, 1) << " top k must not be less than 1.";
  CHECK_LE(top_k_, bottom[0]->channels())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
}

template <typename Dtype>
void MapAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  (*top)[0]->Reshape(1, 2, 1, 1);   //global -- average
}

template <typename Dtype>
void MapAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();

  int valid_num = 0;

  vector<Dtype> label_num(channels, 0);
  vector<Dtype> accuracy_num(channels, 0);
  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < spatial_dim; k++) {
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int j = 0; j < channels; ++j) {
        bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + j * spatial_dim + k], j));
      }
      std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // ------------------------
      int lb = static_cast<int>(bottom_label[i * spatial_dim + k]);
      if ( lb < 0 || lb >= channels )
          continue;
      ++valid_num;

      // check if true label is in top k predictions
      for (int t = 0; t < top_k_; t++) {
        if (bottom_data_vector[t].second == static_cast<int>(bottom_label[i * spatial_dim + k])) {
          ++accuracy_num[ static_cast<int>(bottom_label[i * spatial_dim + k]) ];
          ++accuracy;
          break;
        }
      }
      ++label_num[ static_cast<int>(bottom_label[i * spatial_dim + k]) ];
    }
  }

  Dtype sum = 0.;
  for (int i = 0; i < channels; ++i) {
    LOG(INFO) << i << ": " << accuracy_num[i] << " / " << label_num[i] << " = " << accuracy_num[i] / label_num[i];
    if ( label_num[i] > 0) {
      sum += accuracy_num[i] / label_num[i];
    }
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / (num * spatial_dim);
  (*top)[0]->mutable_cpu_data()[1] = sum / channels;
  LOG(INFO) << "Global Accuracy: " << (*top)[0]->mutable_cpu_data()[0] << "   Average Accuracy: " << (*top)[0]->mutable_cpu_data()[1];
  //LOG(INFO) << "valid_num: " << valid_num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MapAccuracyLayer);

}  // namespace caffe
