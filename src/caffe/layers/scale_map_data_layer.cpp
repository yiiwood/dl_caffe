#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
#include <time.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ScaleMapDataLayer<Dtype>::~ScaleMapDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ScaleMapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const string& source = this->layer_param_.scale_map_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string imgfilename, labfilename;
  while (infile >> imgfilename >> labfilename) {
    lines_.push_back(std::make_pair(imgfilename, labfilename));
  }
  infile.close();
  
  if (this->layer_param_.scale_map_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.scale_map_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.scale_map_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // 
  const int batch_size = this->layer_param_.scale_map_data_param().batch_size();
  scale_size_ = this->layer_param_.scale_map_data_param().scale_size();
  receptive_field_ = this->layer_param_.scale_map_data_param().receptive_field();
  receptive_field_thres_ = int(receptive_field_ * this->layer_param_.scale_map_data_param().receptive_field_thres_ratio());
  receptive_field_labelr_ = this->layer_param_.scale_map_data_param().receptive_field_labelr();
  CHECK(receptive_field_ % 2);
  pad_ = receptive_field_ / 2;
  stride_ = this->layer_param_.scale_map_data_param().stride();
  start_h_ = this->layer_param_.scale_map_data_param().start_h();
  start_w_ = this->layer_param_.scale_map_data_param().start_w();
  if(start_h_ >= 0 && start_h_ < stride_ && start_w_ >= 0 && start_w_ < stride_) {
    start_random_ = false;
    LOG(INFO) << "start_h_ = " << start_h_ << ", start_w_ = " << start_w_;
  } else {
    start_random_ = true;
    LOG(INFO) << "start_h_, start_w_ is random";
  }
  
  bool is_color = true;
  int num_channels = (is_color ? 3 : 1);
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img = cv::imread(lines_[0].first, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << lines_[0].first;
  }

  int image_height = cv_img.rows;
  int image_width = cv_img.cols;
  resize_height_ = int(image_height * scale_size_) - int(image_height * scale_size_) % stride_ + 1;
  resize_width_ = int(image_width * scale_size_) - int(image_width * scale_size_) % stride_ + 1;
  LOG(INFO) << "output valid size: " << resize_height_ << "," << resize_width_;
  // datum size
  this->datum_channels_ = num_channels;
  this->datum_height_ = resize_height_ - stride_ + pad_ * 2;
  this->datum_width_ = resize_width_ - stride_ + pad_ * 2;
  this->datum_size_ = this->datum_channels_ * this->datum_height_ * this->datum_width_;

  // label size
  this->label_channels_ = 1;
  this->label_height_ = resize_height_ / stride_;
  this->label_width_ = resize_width_ / stride_;
  this->label_size_ = this->label_channels_ * this->label_height_ * this->label_width_;

  (*top)[0]->Reshape(batch_size, num_channels, this->datum_height_, this->datum_width_);
  this->prefetch_data_.Reshape(batch_size, num_channels, this->datum_height_, this->datum_width_);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // label
  (*top)[1]->Reshape(batch_size, 1, this->label_height_, this->label_width_);
  this->prefetch_label_.Reshape(batch_size, 1, this->label_height_, this->label_width_);
  LOG(INFO) << "output label size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();
}

template <typename Dtype>
void ScaleMapDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ScaleMapDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ImageDataParameter scale_map_data_param = this->layer_param_.scale_map_data_param();
  const int batch_size = scale_map_data_param.batch_size();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    cv::Mat image_mat;
    //LOG(INFO) << lines_[lines_id_].first;
    ReadData(lines_[lines_id_].first, this->mean_, resize_height_, resize_width_, pad_, image_mat);
    
    int start_h, start_w;
    if(start_random_) {
      start_h = caffe_rng_rand() % stride_;
      start_w = caffe_rng_rand() % stride_;
    } else {
      start_h = start_h_;
      start_w = start_w_;
    }

    cv::Mat label_mat;
    //LOG(INFO) << lines_[lines_id_].second;
    ReadLabel(lines_[lines_id_].second, resize_height_, resize_width_, pad_, 
      receptive_field_, receptive_field_thres_, receptive_field_labelr_, label_mat);
    
    cv::Mat image_data = image_mat(cv::Range(start_h, start_h + this->datum_height_), 
      cv::Range(start_w, start_w + this->datum_width_) );
    cv::Mat label_data;
    cv::resize( label_mat(cv::Range(start_h, start_h + this->datum_height_), 
      cv::Range(start_w, start_w + this->datum_width_) ), 
    label_data, cv::Size(this->label_height_, this->label_width_), 0, 0, CV_INTER_NN);
    
    FillDataLabel(image_data, label_data, item_id, top_data, top_label);
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.scale_map_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
} 
template <typename Dtype>
bool ScaleMapDataLayer<Dtype>::ReadData(string image_path, const Dtype* mean, 
  int rs_height, int rs_width, int pad, cv::Mat &image_mat) {
  int cv_read_flag = CV_LOAD_IMAGE_COLOR;
  cv::Mat cv_img = cv::imread(image_path, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << image_path;
  }

  cv::Mat data_mat = cv::Mat::zeros(cv_img.rows, cv_img.cols, CV_32FC3);
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < data_mat.rows; ++h) {
      for (int w = 0; w < data_mat.cols; ++w) {
        int idx = (c * data_mat.rows + h) * data_mat.cols + w;
        float datum_element = static_cast<float>( static_cast<uint8_t>( cv_img.at<cv::Vec3b>(h, w)[c] ) );
        data_mat.at<cv::Vec3f>(h, w)[c] = (datum_element - mean[idx]);
      }   
    }   
  }
  cv::Mat rs_mat;
  cv::resize(data_mat, rs_mat, cv::Size(rs_width, rs_height));

  int borderType = cv::BORDER_REFLECT;
  cv::copyMakeBorder(rs_mat, image_mat, pad, pad, pad, pad, borderType);
  return true;
}

template <typename Dtype>
bool ScaleMapDataLayer<Dtype>::ReadLabel(string label_path, int rs_height, int rs_width, int pad, 
  int receptive_field, int receptive_field_thres, int receptive_field_labelr, cv::Mat &label_mat) {
  label_mat.create(rs_height + 2*pad, rs_width + 2*pad, CV_32SC1);
  label_mat.setTo(cv::Scalar(0));

  std::ifstream inf(label_path.c_str());
  string imgfn;
  int imgwidth, imgheight;
  int bleft, btop, bwidth, bheight, bcls;
  inf >> imgfn >> imgwidth >> imgheight;
  float scale_w = rs_width * 1.0 / imgwidth;
  float scale_h = rs_height * 1.0 / imgheight;
  while (inf >> bleft >> btop >> bwidth >> bheight >> bcls) {
    float bcw = bleft + bwidth * 1.0 / 2;
    float bch = btop + bheight * 1.0 / 2;

    int rs_bcw = int(bcw * scale_w + 0.5);
    int rs_bch = int(bch * scale_h + 0.5);
    float rs_bwidth = int(bwidth * scale_w + 0.5);
    float rs_bheight = int(bheight * scale_h + 0.5);

    // square box
    int rs_rb;
    if(rs_bwidth < rs_bheight) {
      rs_rb = int(rs_bheight / 2 + 0.5);
    } else {
      rs_rb = int(rs_bwidth / 2 + 0.5);
    }

    /// all zero??? discard??? 
    if( abs(rs_rb * 2 + 1 - receptive_field) < receptive_field_thres ) {
      for(int h = -receptive_field_labelr; h <= receptive_field_labelr; ++h) {
        for(int w = -receptive_field_labelr; w <= receptive_field_labelr; ++w) {
          label_mat.at<int>(pad + rs_bch + h, pad + rs_bcw + w) = 1; 
        }
      }
    }

  }
  inf.close();
  return true;
}

template <typename Dtype>
bool ScaleMapDataLayer<Dtype>::FillDataLabel(cv::Mat &image_data, cv::Mat &label_data, 
  int item_id, Dtype *top_data, Dtype*top_label) {
  for(int c = 0; c < 3; ++c) {
    for(int h = 0; h < image_data.rows; ++h) {
      for(int w = 0; w < image_data.cols; ++w) {
        int idx = ((item_id * 3 + c) * image_data.rows + h) * image_data.cols + w;
        top_data[idx] = image_data.at<cv::Vec3f>(h, w)[c];
      }
    }
  }

  for(int h = 0; h < label_data.rows; ++h) {
    for(int w = 0; w < label_data.cols; ++w) {
      int idx = (item_id * label_data.rows + h) * label_data.cols + w;
      top_label[idx] = label_data.at<int>(h, w);
    }
  }
  return true;
}

INSTANTIATE_CLASS(ScaleMapDataLayer);

}  // namespace caffe
