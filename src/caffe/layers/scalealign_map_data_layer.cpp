#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template <typename Dtype>
ScaleAlignMapDataLayer<Dtype>::~ScaleAlignMapDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.scalealign_map_data_param().backend()) {
  case MapDataParameter_DB_LEVELDB:
    break;  // do nothing
  case MapDataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void ScaleAlignMapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.scalealign_map_data_param().backend()) {
  case MapDataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.scalealign_map_data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.scalealign_map_data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.scalealign_map_data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case MapDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.scalealign_map_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.scalealign_map_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.scalealign_map_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.scalealign_map_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.scalealign_map_data_param().backend()) {
      case MapDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case MapDataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  DatumMap datum_map;
  switch (this->layer_param_.scalealign_map_data_param().backend()) {
  case MapDataParameter_DB_LEVELDB:
    datum_map.ParseFromString(iter_->value().ToString());
    break;
  case MapDataParameter_DB_LMDB:
    datum_map.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    LOG(FATAL) << "Currently crop is not supported in MapDataLayer!";
  }

  ref_depth_ = this->layer_param_.scalealign_map_data_param().ref_depth();
  scalealign_ = this->layer_param_.scalealign_map_data_param().scalealign();
  class_num_ = this->layer_param_.scalealign_map_data_param().class_num();
  sigma_pos_ = this->layer_param_.scalealign_map_data_param().sigma_pos();
  sigma_neg_ = this->layer_param_.scalealign_map_data_param().sigma_neg();

  // datum size
  this->datum_channels_ = datum_map.datum_channels();
  this->datum_height_ = int(datum_map.datum_height() * scalealign_) 
                      - int(datum_map.datum_height() * scalealign_) % 4;
  this->datum_width_ = int(datum_map.datum_width() * scalealign_) 
                     - int(datum_map.datum_width() * scalealign_) % 4;
  this->datum_size_ = this->datum_channels_ * this->datum_height_ * this->datum_width_;

  CHECK_EQ(datum_map.label_channels(), 2);
  // label size
  this->label_channels_ = class_num_;
  this->label_height_ = this->datum_height_ / 4;
  this->label_width_ = this->datum_width_ / 4;
  this->label_size_ = class_num_ * this->label_height_ * this->label_width_;

  (*top)[0]->Reshape(this->layer_param_.scalealign_map_data_param().batch_size(), 
    this->datum_channels_, this->datum_height_, this->datum_width_);
  this->prefetch_data_.Reshape(this->layer_param_.scalealign_map_data_param().batch_size(),
      this->datum_channels_, this->datum_height_, this->datum_width_);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.scalealign_map_data_param().batch_size(), 
      class_num_, this->label_height_, this->label_width_);
    this->prefetch_label_.Reshape(this->layer_param_.scalealign_map_data_param().batch_size(),
        class_num_, this->label_height_, this->label_width_);
    LOG(INFO) << "output label size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();
  }

}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ScaleAlignMapDataLayer<Dtype>::InternalThreadEntry() {
  DatumMap datum_map;
  CHECK(this->prefetch_data_.count());
  Dtype* filled_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* filled_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    filled_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.scalealign_map_data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.scalealign_map_data_param().backend()) {
    case MapDataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum_map.ParseFromString(iter_->value().ToString());
      break;
    case MapDataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum_map.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // fill data and label
    ScaleFill(item_id, datum_map, this->mean_, filled_data, filled_label);

    // go to the next iter
    switch (this->layer_param_.scalealign_map_data_param().backend()) {
    case MapDataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case MapDataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

template<typename Dtype>
void ScaleAlignMapDataLayer<Dtype>::ScaleFill(const int batch_item_id,
                                  const DatumMap& datum_map,
                                  const Dtype* mean,
                                  Dtype* filled_data, Dtype* filled_label) {
  const string& data = datum_map.data();
  const Dtype scale = this->layer_param_.transform_param().scale();
  int channels = datum_map.datum_channels();
  if (channels != 3) {
      LOG(FATAL) << "The channels must be 3!";
  }
  CHECK_EQ(datum_map.label_channels(), 2);

  cv::Mat data_mat = cv::Mat::zeros(datum_map.datum_height(), datum_map.datum_width(), CV_32FC3);
  if (data.size()) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < data_mat.rows; ++h) {
        for (int w = 0; w < data_mat.cols; ++w) {
          int idx = (c * data_mat.rows + h) * data_mat.cols + w;
          Dtype datum_element = static_cast<float>(static_cast<uint8_t>(data[idx]));
          data_mat.at<cv::Vec3b>(h, w)[c] = (datum_element - mean[idx]) * scale;
        }
      }
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < data_mat.rows; ++h) {
        for (int w = 0; w < data_mat.cols; ++w) {
          int idx = (c * data_mat.rows + h) * data_mat.cols + w;
          data_mat.at<cv::Vec3b>(h, w)[c] = (datum_map.float_data(idx) - mean[idx]) * scale;
        }
      }
    }
  }

  cv::Mat scalealign_data;
  cv::resize(data_mat, scalealign_data, cv::Size(this->datum_height_, this->datum_width_));
  // fill
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < scalealign_data.rows; ++h) {
      for (int w = 0; w < scalealign_data.cols; ++w) {
        int idx = (c * scalealign_data.rows + h) * scalealign_data.cols + w;
        filled_data[batch_item_id * this->datum_size_ + idx] = scalealign_data.at<cv::Vec3b>(h, w)[c];
      }
    }
  }

  cv::Mat label_mat = cv::Mat::zeros(datum_map.label_height(), datum_map.label_width(), CV_32SC1);
  cv::Mat depth_mat = cv::Mat::zeros(datum_map.label_height(), datum_map.label_width(), CV_32FC1);
  int size = datum_map.label_height() * datum_map.label_width();
  for (int h = 0; h < label_mat.rows; ++h) {
    for (int w = 0; w < label_mat.cols; ++w) {
      int idx = h * label_mat.cols + w;
      label_mat.at<int>(h, w) = static_cast<int>(datum_map.label_map(idx) );
      depth_mat.at<float>(h, w) = static_cast<float>(datum_map.label_map(size + idx) );
    }
  }
  cv::Mat scalealign_label, scalealign_depth;
  cv::resize(label_mat, scalealign_label, cv::Size(this->label_height_, this->label_width_), 0, 0, CV_INTER_NN);
  cv::resize(depth_mat, scalealign_depth, cv::Size(this->label_height_, this->label_width_), 0, 0, CV_INTER_NN);

  int scalealign_size = this->label_height_* this->label_width_;
  for (int h = 0; h < scalealign_label.rows; ++h) {
    for (int w = 0; w < scalealign_label.cols; ++w) {
      int idx = h * scalealign_label.cols + w;
      int label = scalealign_label.at<int>(h, w);
      float depth = scalealign_depth.at<float>(h, w);
      float max_ratio = (depth / (scalealign_ * ref_depth_)) > ((scalealign_ * ref_depth_) / depth )?
            (depth / (scalealign_ * ref_depth_)) : ((scalealign_ * ref_depth_) / depth );
      if (label < 0 || label >= class_num_ || (max_ratio > sigma_pos_ && max_ratio < sigma_neg_) ) {
        for (int c = 0; c < class_num_; ++c) {
          filled_label[batch_item_id * this->label_size_ + c * scalealign_size + idx] = -1;
        }
      }
      else {
        for (int c = 0; c < class_num_; ++c) {
          filled_label[batch_item_id * this->label_size_ + c * scalealign_size + idx] = 0;
        }
        if (max_ratio <= sigma_pos_) {
          filled_label[batch_item_id * this->label_size_ + label * scalealign_size + idx] = 1;
        }
      } 
    }
  }
  
}


INSTANTIATE_CLASS(ScaleAlignMapDataLayer);

}  // namespace caffe
