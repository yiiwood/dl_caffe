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
PositionMapDataLayer<Dtype>::~PositionMapDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.map_data_param().backend()) {
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
void PositionMapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.map_data_param().backend()) {
  case MapDataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.map_data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.map_data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.map_data_param().source() << std::endl
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
             this->layer_param_.map_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.map_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.map_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.map_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.map_data_param().backend()) {
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
  switch (this->layer_param_.map_data_param().backend()) {
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
  jitter_ = this->layer_param_.map_data_param().jitter();
  CHECK(!jitter_) << "PositionMapDataLayer does not suppuort jitter!";

  (*top)[0]->Reshape(
      this->layer_param_.map_data_param().batch_size(), datum_map.datum_channels() + 2,
      datum_map.datum_height(), datum_map.datum_width());
  this->prefetch_data_.Reshape(this->layer_param_.map_data_param().batch_size(),
      datum_map.datum_channels(), datum_map.datum_height(), datum_map.datum_width());
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(
        this->layer_param_.map_data_param().batch_size(), datum_map.label_channels(),
        datum_map.label_height(), datum_map.label_width());
    this->prefetch_label_.Reshape(this->layer_param_.map_data_param().batch_size(),
        datum_map.label_channels(), datum_map.label_height(), datum_map.label_width());
    LOG(INFO) << "output label size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();
  }
  // datum size
  this->datum_channels_ = datum_map.datum_channels();
  this->datum_height_ = datum_map.datum_height();
  this->datum_width_ = datum_map.datum_width();
  this->datum_size_ = datum_map.datum_channels() * datum_map.datum_height() * datum_map.datum_width();

  // label size
  this->label_channels_ = datum_map.label_channels();
  this->label_height_ = datum_map.label_height();
  this->label_width_ = datum_map.label_width();
  this->label_size_ = datum_map.label_channels() * datum_map.label_height() * datum_map.label_width();

  ///======================== position ============================///
  pos_blob = new Blob<Dtype>(1, 2, datum_map.datum_height(), datum_map.datum_width()); 
  Dtype * pos_data = pos_blob->mutable_cpu_data();
  for(int h = 0; h < this->datum_height_; ++h) {
    for(int w = 0; w < this->datum_width_; ++w) {
      pos_data[h * this->datum_width_ + w] = this->datum_height_ / 2.0 - h;
    }
  }
  for(int h = 0; h < this->datum_height_; ++h) {
    for(int w = 0; w < this->datum_width_; ++w) {
      pos_data[this->datum_height_ * this->datum_width_ + h * this->datum_width_ + w] = w - this->datum_width_ / 2.0;
    }
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void PositionMapDataLayer<Dtype>::InternalThreadEntry() {
  DatumMap datum_map;
  CHECK(this->prefetch_data_.count());
  Dtype* filled_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* filled_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    filled_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.map_data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.map_data_param().backend()) {
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
    this->data_transformer_.Fill(item_id, datum_map, this->mean_, filled_data, filled_label);

    // go to the next iter
    switch (this->layer_param_.map_data_param().backend()) {
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

template <typename Dtype>
void PositionMapDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  this->JoinPrefetchThread();
  int num = this->prefetch_data_.num();
  int dim = this->prefetch_data_.count() / num;
  int pdim = 2 * this->prefetch_data_.height() * this->prefetch_data_.width();
  for(int i = 0; i < num; ++i) {
    caffe_copy(dim, this->prefetch_data_.cpu_data() + i * dim,(*top)[0]->mutable_cpu_data() + i * (dim + pdim));
    caffe_copy(pdim, pos_blob->cpu_data(),(*top)[0]->mutable_cpu_data() + i * (dim + pdim) + dim);
  }
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
          (*top)[1]->mutable_cpu_data());
  }
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(PositionMapDataLayer);

}  // namespace caffe
