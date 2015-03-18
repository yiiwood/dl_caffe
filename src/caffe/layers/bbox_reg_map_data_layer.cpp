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
BBoxRegMapDataLayer<Dtype>::~BBoxRegMapDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  delete prefetch_bbox_;
}

template <typename Dtype>
void BBoxRegMapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const string& source = this->layer_param_.bbox_reg_map_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string imgfilename, labfilename;
  while (infile >> imgfilename >> labfilename) {
    lines_.push_back(std::make_pair(imgfilename, labfilename));
  }
  infile.close();
  
  if (this->layer_param_.bbox_reg_map_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.bbox_reg_map_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.bbox_reg_map_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // 
  const int batch_size = this->layer_param_.bbox_reg_map_data_param().batch_size();
  stride_ = this->layer_param_.bbox_reg_map_data_param().stride();
  
  bool is_color = true;
  int num_channels = (is_color ? 3 : 1);
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img = cv::imread(lines_[0].first, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << lines_[0].first;
  }

  int image_height = cv_img.rows;
  int image_width = cv_img.cols;
  int resize_height_ = image_height - image_height % stride_;
  int resize_width_ = image_width  - image_width % stride_;
  LOG(INFO) << "output valid size: " << resize_height_ << "," << resize_width_;
  // datum size
  this->datum_channels_ = num_channels;
  this->datum_height_ = resize_height_;
  this->datum_width_ = resize_width_;
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

  // normalized bbox
  prefetch_bbox_ = new Blob<Dtype>(batch_size, 4, this->label_height_, this->label_width_);
  (*top)[2]->Reshape(batch_size, 4, this->label_height_, this->label_width_);
  LOG(INFO) << "output bbox size: " << (*top)[2]->num() << ","
      << (*top)[2]->channels() << "," << (*top)[2]->height() << ","
      << (*top)[2]->width();
}

template <typename Dtype>
void BBoxRegMapDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void BBoxRegMapDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* top_bbox = prefetch_bbox_->mutable_cpu_data();
  ImageDataParameter bbox_reg_map_data_param = this->layer_param_.scale_map_data_param();
  const int batch_size = bbox_reg_map_data_param.batch_size();

  int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    cv::Mat image_mat;
    ReadData(lines_[lines_id_].first, this->mean_, resize_height_, resize_width_, image_mat);

    cv::Mat label_mat, bbox_mat;
    ReadLabelBBox(lines_[lines_id_].second, resize_height_, resize_width_, label_mat, bbox_mat);
    
    cv::Mat label_data;
    cv::resize( label_mat, label_data, cv::Size(this->label_height_, this->label_width_), 0, 0, CV_INTER_NN);
    cv::Mat bbox_data;
    cv::resize( bbox_mat, bbox_data, cv::Size(this->label_height_, this->label_width_), 0, 0, CV_INTER_NN);
    
    FillDataLabelBBox(image_mat, label_data, bbox_data, item_id, top_data, top_label, top_bbox);
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.bbox_reg_map_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
} 

template <typename Dtype>
bool BBoxRegMapDataLayer<Dtype>::ReadData(string image_path, const Dtype* mean, 
  int rs_height, int rs_width, cv::Mat &image_mat) {
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
  
  cv::resize(data_mat, image_mat, cv::Size(rs_width, rs_height));

  return true;
}

template <typename Dtype>
bool BBoxRegMapDataLayer<Dtype>::ReadLabelBBox(string label_path, int rs_height, int rs_width, 
        cv::Mat &label_mat, cv::Mat &bbox_mat) {
  label_mat.create(rs_height, rs_width, CV_32SC1);
  label_mat.setTo(cv::Scalar(0));

  bbox_mat.create(rs_height, rs_width, CV_32FC4);
  bbox_mat.setTo(cv::Scalar(0.0, 0.0, 0.0, 0.0));

  std::ifstream inf(label_path.c_str());
  string imgfn;
  int imgwidth, imgheight;
  int bleft, btop, bwidth, bheight, bcls;
  inf >> imgfn >> imgwidth >> imgheight;
  float scale_w = rs_width * 1.0 / imgwidth;
  float scale_h = rs_height * 1.0 / imgheight;
  while (inf >> bleft >> btop >> bwidth >> bheight >> bcls) {
    int rs_bleft = int(bleft * scale_w + 0.5);
    int rs_btop = int(btop * scale_h + 0.5);
    int rs_bwidth = int(bwidth * scale_w + 0.5);
    int rs_bheight = int(bheight * scale_h + 0.5);
    float n_rs_bleft = rs_bleft * 1.0 / rs_width;
    float n_rs_btop = rs_btop * 1.0 / rs_height;
    float n_rs_bright = (rs_bleft + rs_bwidth) * 1.0 / rs_width;
    float n_rs_bbottom = (rs_btop + rs_bheight) * 1.0 / rs_height;

    /// 
    for(int h = rs_btop; h < rs_btop + rs_bheight; ++h) {
      for(int w = rs_bleft; w < rs_bleft + rs_bwidth; ++w) {
        label_mat.at<int>(h, w) = 1;
        bbox_mat.at<cv::Vec4f>(h, w)[0] = n_rs_btop - h * 1.0 / rs_height;
        bbox_mat.at<cv::Vec4f>(h, w)[1] = n_rs_bleft - w * 1.0 / rs_width;
        bbox_mat.at<cv::Vec4f>(h, w)[2] = n_rs_bbottom - h * 1.0 / rs_height;
        bbox_mat.at<cv::Vec4f>(h, w)[3] = n_rs_bright - w * 1.0 / rs_width;
      }
    }

  }
  inf.close();
  return true;
}

template <typename Dtype>
bool BBoxRegMapDataLayer<Dtype>::FillDataLabelBBox(cv::Mat &image_data, cv::Mat &label_data, cv::Mat &bbox_data, 
  int item_id, Dtype *top_data, Dtype *top_label, Dtype *top_bbox) {
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

  for(int c = 0; c < 4; ++c) {
    for(int h = 0; h < bbox_data.rows; ++h) {
      for(int w = 0; w < bbox_data.cols; ++w) {
        int idx = ((item_id * 4 + c) * bbox_data.rows + h) * bbox_data.cols + w;
        top_bbox[idx] = bbox_data.at<cv::Vec4f>(h, w)[c];
      }   
    }   
  }
  return true;
}

template <typename Dtype>
void BBoxRegMapDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  this->JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
       (*top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        (*top)[1]->mutable_cpu_data());
    caffe_copy(prefetch_bbox_->count(), prefetch_bbox_->cpu_data(),
        (*top)[2]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(BBoxRegMapDataLayer);

}  // namespace caffe
