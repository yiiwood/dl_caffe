#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"
#include "opencv2/opencv.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  virtual ~BaseDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}

  int datum_channels() const { return datum_channels_; }
  int datum_height() const { return datum_height_; }
  int datum_width() const { return datum_width_; }
  int datum_size() const { return datum_size_; }

 protected:
  TransformationParameter transform_param_;
  DataTransformer<Dtype> data_transformer_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  Blob<Dtype> data_mean_;
  const Dtype* mean_;
  Caffe::Phase phase_;
  bool output_labels_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual ~BasePrefetchingDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  // The thread's function
  virtual void InternalThreadEntry() {}

 protected:
  Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_label_;
};

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void InternalThreadEntry();

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

template <typename Dtype>
class MapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MAP_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

 protected:
  virtual void InternalThreadEntry();

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  bool jitter_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

template <typename Dtype>
class PositionMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PositionMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~PositionMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
     
  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_POSITION_MAP_DATA;
  }
    
  virtual inline int ExactNumBottomBlobs() const { return 0; } 
  virtual inline int MinTopBlobs() const { return 1; } 
  virtual inline int MaxTopBlobs() const { return 2; } 
         
  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void InternalThreadEntry();
     
  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  bool jitter_;
               
  Blob<Dtype> *pos_blob;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};


template <typename Dtype>
class ScaleMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ScaleMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ScaleMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SCALE_MAP_DATA;
  }

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
    
  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  bool ReadData(string image_path, const Dtype* mean, int rs_height, int rs_width, int pad, cv::Mat &image_mat);
  bool ReadLabel(string label_path, int rs_height, int rs_width, int pad, 
    int receptive_field, int receptive_field_thres, int receptive_field_labelr, cv::Mat &label_mat);
  bool FillDataLabel(cv::Mat &image_data, cv::Mat &label_data, int item_id, Dtype *top_data, Dtype*top_label);

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  bool jitter_;

  float scale_size_;
  int resize_height_;
  int resize_width_;
  int receptive_field_;
  int receptive_field_thres_;
  int receptive_field_labelr_;

  int pad_;
  int stride_;
  int start_h_;
  int start_w_;
  bool start_random_;

  vector<std::pair<std::string, string> > lines_;
  int lines_id_;
  
};

template <typename Dtype>
class BBoxRegMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BBoxRegMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~BBoxRegMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_BBOX_REG_MAP_DATA;
  }

  virtual inline int ExactNumBottomBlobs() const { return 0; } 
  virtual inline int MinTopBlobs() const { return 1; } 
  virtual inline int MaxTopBlobs() const { return 3; } 
    
  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  bool ReadData(string image_path, const Dtype* mean, int rs_height, int rs_width, cv::Mat &image_mat);
  bool ReadLabelBBox(string label_path, int rs_height, int rs_width, cv::Mat &label_mat, cv::Mat &bbox_mat);
  bool FillDataLabelBBox(cv::Mat &image_data, cv::Mat &label_data, cv::Mat &bbox_data, 
          int item_id, Dtype *top_data, Dtype *top_label, Dtype *top_bbox);

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;

  int resize_height_;
  int resize_width_;
  int stride_;
  Blob<Dtype>* prefetch_bbox_;
  vector<std::pair<std::string, string> > lines_;
  int lines_id_;
};

template <typename Dtype>
class MultiScaleMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultiScaleMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultiScaleMapDataLayer();
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MULTI_SCALE_MAP_DATA;
  }

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
    
  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  bool ReadData(string image_path, const Dtype* mean, int rs_height, int rs_width, int pad, cv::Mat &image_mat);
  bool ReadLabel_IOU(string label_path, int rs_height, int rs_width, int pad, 
    int receptive_field, float min_iou, cv::Mat &label_mat);
  bool ReadLabel_CENT(string label_path, int rs_height, int rs_width, int pad,
    int receptive_field, int receptive_field_thres, int receptive_field_labelr, cv::Mat &label_mat);
  float IOU(int top1, int left1, int bottom1, int right1, 
            int top2, int left2, int bottom2, int right2);
  bool FillDataLabel(cv::Mat &image_data, cv::Mat &label_data, int item_id, Dtype *top_data, Dtype*top_label);
  void DataReshape(vector<Blob<Dtype>*>* top, float scale);

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  bool jitter_;

  int resize_height_;
  int resize_width_;
  int receptive_field_;
  float min_iou_;

  int num_channels_;
  int image_height_;
  int image_width_;

  vector<Blob<Dtype>*>* data_top;
  int pad_;
  int stride_;
  int start_h_;
  int start_w_;
  bool start_random_;
  vector<float> scale_vector_;
  int scale_id_;

  vector<std::pair<std::string, string> > lines_;
  int start_lines_id_;
  int lines_id_;

};

template <typename Dtype>
class MultiScaleMaskMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultiScaleMaskMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultiScaleMaskMapDataLayer();
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MULTI_SCALE_MASK_MAP_DATA;
  }

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
    
  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  bool ReadData(string image_path, const Dtype* mean, int rs_height, int rs_width, int pad, cv::Mat &image_mat);
  bool ReadLabel_IOU_Mask(string label_path, int rs_height, int rs_width, int pad,
    int receptive_field, float neg_max_iou, float min_iou, cv::Mat &label_mat);
  bool ReadLabel_CENT_Mask(string label_path, int rs_height, int rs_width, int pad,
    int receptive_field, int receptive_field_thres, int receptive_field_labelr, cv::Mat &label_mat);
  float IOU(int top1, int left1, int bottom1, int right1,
            int top2, int left2, int bottom2, int right2);
  bool FillDataLabel(cv::Mat &image_data, cv::Mat &label_data, int item_id, Dtype *top_data, Dtype*top_label);
  void DataReshape(vector<Blob<Dtype>*>* top, float scale);

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  bool jitter_;

  int resize_height_;
  int resize_width_;
  int receptive_field_;
  float min_iou_;
  float neg_max_iou_;
  bool bbox_mask_;

  int num_channels_;
  int image_height_;
  int image_width_;

  vector<Blob<Dtype>*>* data_top;
  int pad_;
  int stride_;
  int start_h_;
  int start_w_;
  bool start_random_;
  vector<float> scale_vector_;
  int scale_id_;

  vector<std::pair<std::string, string> > lines_;
  int start_lines_id_;
  int lines_id_;

};

template <typename Dtype>
class MultiScaleCropMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultiScaleCropMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultiScaleCropMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MULTI_SCALE_CROP_MAP_DATA;
  }

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
    
  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  bool ReadData(string image_path, const Dtype* mean, int rs_height, int rs_width, int pad, cv::Mat &image_mat);
  int ReadLabel_CENT_Mask(string label_path, int rs_height, int rs_width, int pad,
    int receptive_field, int receptive_field_thres, int receptive_field_labelr, cv::Mat &label_mat);
  bool FillDataLabel(cv::Mat &image_data, cv::Mat &label_data, int item_id, Dtype *top_data, Dtype*top_label);

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  bool jitter_;

  int resize_height_;
  int resize_width_;
  int receptive_field_;
  int crop_size_;

  int num_channels_;
  int image_height_;
  int image_width_;

  int pad_;
  int stride_;
  bool start_random_;
  vector<float> scale_vector_;
  vector<int> scale_smpnum_;
  int base_smpnum;
  int scale_id_;

  vector<std::pair<std::string, string> > lines_;
  int start_lines_id_;
  int lines_id_;

};

template <typename Dtype>
class PyramidMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PyramidMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~PyramidMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
      return LayerParameter_LayerType_PYRAMID_MAP_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 6; }

  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void InternalThreadEntry();

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;

  int level_num_;
  vector<Blob<Dtype>* > prefetch_pyramid_data_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

template <typename Dtype>
class PyramidRecurrentMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PyramidRecurrentMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~PyramidRecurrentMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_PYRAMID_RECURRENT_MAP_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 6; }

  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void InternalThreadEntry();

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;

  int level_num_;
  vector<Blob<Dtype>* > prefetch_pyramid_data_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

template <typename Dtype>
class PatchDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PatchDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~PatchDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
      return LayerParameter_LayerType_PATCH_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 6; }
  
  void FillMat(const DatumMap& datum_map, const Dtype* mean, 
        cv::Mat &image_data, cv::Mat &image_label);
  void SamplingPatch(vector<cv::Mat*> &prefetch_image_data, 
        vector<cv::Mat*> &prefetch_image_label, const int class_num, 
        const bool mirror, const float scale_range, const float rotate_range, 
        Blob<Dtype> &prefetch_data, Blob<Dtype> &prefetch_label);
  void FillPatch(cv::Mat &image_data, cv::Mat &image_label, 
        const int h, const int w, const int patch_size,
        const bool mirror, const float scale_range, const float rotate_range,
        Dtype *prefetch_data, Dtype *prefetch_label);

 protected:
  virtual void InternalThreadEntry();

  bool jitter_;
  int image_size_;
  int patch_size_;
  int class_num_;

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;

  vector<cv::Mat*> prefetch_image_data_;
  vector<cv::Mat*> prefetch_image_label_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

template <typename Dtype>
class DepthMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DepthMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~DepthMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_DEPTH_MAP_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 6; }

  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  int depth_channels() const { return depth_channels_; }
  int depth_height() const { return depth_height_; }
  int depth_width() const { return depth_width_; }
  int depth_size() const { return depth_size_; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void InternalThreadEntry();

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  int depth_channels_;
  int depth_height_;
  int depth_width_;
  int depth_size_;

  Blob<Dtype> prefetch_depth_data_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

template <typename Dtype>
class ScaleAlignMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ScaleAlignMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ScaleAlignMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SCALEALIGN_MAP_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 6; }

  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  void ScaleFill(const int batch_item_id, const DatumMap& datum_map, const Dtype* mean,
      Dtype* filled_data, Dtype* filled_label);

 protected:
  virtual void InternalThreadEntry();

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  float ref_depth_;
  float scalealign_;
  int class_num_;
  float sigma_pos_;
  float sigma_neg_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

template <typename Dtype>
class ContourMapDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ContourMapDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ContourMapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_CONTOUR_MAP_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 6; }

  int label_channels() const { return label_channels_; }
  int label_height() const { return label_height_; }
  int label_width() const { return label_width_; }
  int label_size() const { return label_size_; }

  int contour_channels() const { return contour_channels_; }
  int contour_height() const { return contour_height_; }
  int contour_width() const { return contour_width_; }
  int contour_size() const { return contour_size_; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void InternalThreadEntry();

  int label_channels_;
  int label_height_;
  int label_width_;
  int label_size_;
  int contour_channels_;
  int contour_height_;
  int contour_width_;
  int contour_size_;

  Blob<Dtype> prefetch_contour_data_;

  // LEVELDB
  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

/**
 * @brief Provides data to the Net generated by a Filler.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DummyDataLayer : public Layer<Dtype> {
 public:
  explicit DummyDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_DUMMY_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}

  vector<shared_ptr<Filler<Dtype> > > fillers_;
  vector<bool> refill_;
};

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5DataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_HDF5_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual void LoadHDF5FileData(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

/**
 * @brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param);
  virtual ~HDF5OutputLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_HDF5_OUTPUT;
  }
  // TODO: no limit on the number of blobs
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

  inline std::string file_name() const { return file_name_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void SaveBlobs();

  std::string file_name_;
  hid_t file_id_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_IMAGE_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
};

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MemoryDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MEMORY_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, Dtype* label, int n);

  int batch_size() { return batch_size_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  int batch_size_;
  Dtype* data_;
  Dtype* labels_;
  int n_;
  int pos_;
  Blob<Dtype> added_data_;
  Blob<Dtype> added_label_;
  bool has_new_data_;
};

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_WINDOW_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void InternalThreadEntry();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
