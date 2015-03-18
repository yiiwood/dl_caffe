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

namespace caffe {

template <typename Dtype>
PatchDataLayer<Dtype>::~PatchDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.patch_data_param().backend()) {
  case PatchDataParameter_DB_LEVELDB:
    break;  // do nothing
  case PatchDataParameter_DB_LMDB:
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
void PatchDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.patch_data_param().backend()) {
  case PatchDataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.patch_data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.patch_data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.patch_data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case PatchDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.patch_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.patch_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.patch_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.patch_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.patch_data_param().backend()) {
      case PatchDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case PatchDataParameter_DB_LMDB:
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
  switch (this->layer_param_.patch_data_param().backend()) {
  case PatchDataParameter_DB_LEVELDB:
    datum_map.ParseFromString(iter_->value().ToString());
    break;
  case PatchDataParameter_DB_LMDB:
    datum_map.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
  CHECK_EQ(datum_map.datum_height(), datum_map.label_height());
  CHECK_EQ(datum_map.datum_width(), datum_map.label_width());
  CHECK_EQ(datum_map.datum_channels(), 3);
  CHECK_EQ(datum_map.label_channels(), 1);

  jitter_ = this->layer_param_.patch_data_param().jitter();
  image_size_ = this->layer_param_.patch_data_param().image_size();
  patch_size_ = this->layer_param_.patch_data_param().patch_size();
  CHECK_EQ(patch_size_ % 2, 1);
  class_num_ = this->layer_param_.patch_data_param().class_num();

  // patch
  (*top)[0]->Reshape(
      this->layer_param_.patch_data_param().batch_size(), datum_map.datum_channels(),
      patch_size_, patch_size_);
  this->prefetch_data_.Reshape(this->layer_param_.patch_data_param().batch_size(),
      datum_map.datum_channels(), patch_size_, patch_size_);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.patch_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.patch_data_param().batch_size(), 1, 1, 1);
    LOG(INFO) << "output label size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();
  }

  for (int i = 0; i < image_size_; ++i) {
    cv::Mat *image_data = new cv::Mat(datum_map.datum_height(), datum_map.datum_width(), CV_32FC3);
    prefetch_image_data_.push_back(image_data);

    cv::Mat *image_label = new cv::Mat(datum_map.label_height(), datum_map.label_width(), CV_32SC1);
    prefetch_image_label_.push_back(image_label);
  }

  // datum size
  this->datum_channels_ = datum_map.datum_channels();
  this->datum_height_ = patch_size_;
  this->datum_width_ = patch_size_;
  this->datum_size_ = datum_map.datum_channels() * patch_size_ * patch_size_;

  // label size
  this->label_channels_ = datum_map.label_channels();
  this->label_height_ = 1;
  this->label_width_ = 1;
  this->label_size_ = datum_map.label_channels() * 1 * 1;
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void PatchDataLayer<Dtype>::InternalThreadEntry() {
  DatumMap datum_map;

  for (int image_id = 0; image_id < image_size_; ++image_id) {
    // get a blob
    switch (this->layer_param_.patch_data_param().backend()) {
    case PatchDataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum_map.ParseFromString(iter_->value().ToString());
      break;
    case PatchDataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum_map.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // fill data and label
    FillMat(datum_map, this->mean_, *(prefetch_image_data_[image_id]), *(prefetch_image_label_[image_id]));

    // go to the next iter
    switch (this->layer_param_.patch_data_param().backend()) {
    case PatchDataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case PatchDataParameter_DB_LMDB:
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

  // patch sampling
  CHECK(this->prefetch_data_.count());
  Dtype* filled_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* filled_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    filled_label = this->prefetch_label_.mutable_cpu_data();
  }
  
  if (jitter_) {
    SamplingPatch(prefetch_image_data_, prefetch_image_label_, class_num_, true, 0.05, 10, this->prefetch_data_, this->prefetch_label_);
  }
  else {
    SamplingPatch(prefetch_image_data_, prefetch_image_label_, class_num_, false, 0, 0, this->prefetch_data_, this->prefetch_label_);
  }

}

template <typename Dtype>
void PatchDataLayer<Dtype>::FillMat(const DatumMap& datum_map, const Dtype* mean, 
  cv::Mat &image_data, cv::Mat &image_label) {

  // data
  const Dtype scale = this->transform_param_.scale();
  const string& data = datum_map.data();
  if (data.size()) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < image_data.rows; ++h) {
        for (int w = 0; w < image_data.cols; ++w) {
          int idx = (c * image_data.rows + h) * image_data.cols + w;
          Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[idx]));
          image_data.at<cv::Vec3f>(h, w)[c] = static_cast<float>( (datum_element - mean[idx]) * scale);
        }
      }
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < image_data.rows; ++h) {
        for (int w = 0; w < image_data.cols; ++w) {
          int idx = (c * image_data.rows + h) * image_data.cols + w;
          image_data.at<cv::Vec3f>(h, w)[c] = (datum_map.float_data(idx) - mean[idx]) * scale;
        }
      }
    }
  }

  // label
  for (int h = 0; h < image_label.rows; ++h) {
    for (int w = 0; w < image_label.cols; ++w) {
      int idx = h * image_label.cols + w;
      image_label.at<int>(h, w) = static_cast<int>(datum_map.label_map(idx) );
    }   
  }

}

template <typename Dtype>
void PatchDataLayer<Dtype>::SamplingPatch(vector<cv::Mat*> &prefetch_image_data, 
    vector<cv::Mat*> &prefetch_image_label, const int class_num, 
    const bool mirror, const float scale_range, const float rotate_range, 
    Blob<Dtype> &prefetch_data, Blob<Dtype> &prefetch_label) {

  int height = prefetch_image_data[0]->rows;
  int width = prefetch_image_data[0]->cols;
  int spatial_dim =  height * width;
  int batch_size = prefetch_data.num();
  int patch_size = prefetch_data.height();

  int dim = class_num * spatial_dim;
  int base_num = batch_size / class_num;

  std::vector<std::vector<int> > cls_arr(class_num);
  for (int n = 0; n < prefetch_image_data.size(); ++n) {
    for (int s = 0; s < spatial_dim; ++s) {
      int idx = n * spatial_dim + s;
      int lb = prefetch_image_label[n]->at<int>(s/width, s%width);
      if (lb < 0 || lb >= class_num) continue;

      cls_arr[lb].push_back(idx);
    }
  }

  int accu_num = 0;
  vector<Dtype> sampling_prob(class_num, 1); 
  vector<int> sampling_num(class_num, 0);

  for (int i = 0; i < cls_arr.size(); ++i) {
      sampling_num[i] = cls_arr[i].size() > base_num ? base_num : cls_arr[i].size();
      accu_num += sampling_num[i];
  }

  int i = 0;
  while (accu_num < batch_size) {
    if (cls_arr[i].size() > sampling_num[i]) {
      sampling_num[i]++;
      accu_num++;
    }
    i = (i + 1) % cls_arr.size();
  }
  CHECK_EQ(accu_num, batch_size);

  int bidx = 0;
  for (int k = 0; k < cls_arr.size(); ++i) {
    int smp_num = sampling_num[k];
    int smp_range = cls_arr[k].size();
    for (int j = 0; j < smp_num; ++j) {
      int sidx = j + (caffe::caffe_rng_rand() % smp_range);
      int idx = cls_arr[k][sidx];
      int s = idx % spatial_dim;
      int w = s % width;
      int h = s / height;
      int n = idx / spatial_dim;
      FillPatch( *(prefetch_image_data[n]), *(prefetch_image_label[n]), h, w, patch_size, mirror, scale_range, rotate_range,  
        prefetch_data.mutable_cpu_data() + bidx * 3 * patch_size * patch_size, prefetch_label.mutable_cpu_data() + bidx);
      std::swap(cls_arr[k][sidx], cls_arr[k][j]);
      smp_range--;
      ++bidx;
    }
  }
}

template <typename Dtype>
void PatchDataLayer<Dtype>::FillPatch(cv::Mat &image_data, cv::Mat &image_label, 
    const int h, const int w, const int patch_size,
    const bool mirror, const float scale_range, const float rotate_range,
    Dtype *prefetch_data, Dtype *prefetch_label) {
  
  *prefetch_label = image_label.at<int>(h, w);

  int height = image_data.rows;
  int width = image_data.cols;
  int half_patch_size = patch_size/2;
  float scale_rate = ((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1)) * scale_range;
  int scale_patch_size = static_cast<int>(scale_rate * patch_size) + patch_size;
  if (scale_patch_size % 2 == 0)  scale_patch_size += 1;
  int scale_half_patch_size = scale_patch_size / 2;
  int h_st = (h - scale_half_patch_size) < 0? 0 : (h - scale_half_patch_size);
  int h_ed = (h + scale_half_patch_size) > (height - 1)? (height - 1) : (h + scale_half_patch_size);
  h_ed += 1;
  int w_st = (w - scale_half_patch_size) < 0? 0 : (w - scale_half_patch_size);
  int w_ed = (w + scale_half_patch_size) > (width - 1)? (width - 1) : (w + scale_half_patch_size);
  w_ed += 1;
  int dh_st = scale_half_patch_size - (h - h_st);
  int dh_ed = scale_half_patch_size + (h_ed - h);
  int dw_st = scale_half_patch_size - (w - w_st);
  int dw_ed = scale_half_patch_size + (w_ed - w);

  cv::Mat data_mat = cv::Mat::zeros(scale_patch_size, scale_patch_size, CV_32FC3);
  image_data(cv::Range(h_st, h_ed), cv::Range(w_st, w_ed)).copyTo( 
    data_mat(cv::Range(dh_st, dh_ed), cv::Range(dw_st, dw_ed)) );
  cv::Mat data_patch;
  cv::resize(data_mat, data_patch, cv::Size(patch_size, patch_size));
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < data_patch.rows; ++h) {
      for (int w = 0; w < data_patch.cols; ++w) {
         prefetch_data[(c * patch_size + h) * patch_size + w] = 
              static_cast<Dtype>( data_patch.at<cv::Vec3f>(h, w)[c] ); 
      }   
    }   
  }

}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(PatchDataLayer, Forward);
#endif

INSTANTIATE_CLASS(PatchDataLayer);

}  // namespace caffe
