#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param)
    : param_(param) {
    phase_ = Caffe::phase();
  }
  virtual ~DataTransformer() {}

  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param batch_item_id
   *    Datum position within the batch. This is used to compute the
   *    writing position in the top blob's data
   * @param datum
   *    Datum containing the data to be transformed.
   * @param mean
   * @param transformed_data
   *    This is meant to be the top blob's data. The transformed data will be
   *    written at the appropriate place within the blob's data.
   */
  void Transform(const int batch_item_id, const Datum& datum,
                 const Dtype* mean, Dtype* transformed_data);

  void Jitter(DatumMap& datum_map, const bool mirror,
          const float scale_range, const float rotate_range);

  void FillData(const int batch_item_id, const DatumMap& datum_map, const Dtype* mean,
            const int height, const int width, Dtype* filled_data);
  void FillLabel(const int batch_item_id, const DatumMap& datum_map,
            const int height, const int width, Dtype* filled_label);
  void DatumMapToDataMat(const DatumMap& datum_map, const Dtype* mean, cv::Mat &data_mat);
  void DatumMapToLabelMat(const DatumMap& datum_map, cv::Mat &label_mat);
  
  void Fill(const int batch_item_id, const DatumMap& datum_map,
            const Dtype* mean, Dtype* filled_data, Dtype* filled_label);
 
  void Fill(const int batch_item_id, const Datum& datum,
            const Dtype* mean, Dtype* filled_data);

  void Pyramid(const int batch_item_id, const DatumMap& datum_map,
            const Dtype* mean, vector<Blob<Dtype>* > &pyramid_data, const int level_num);

  void DepthFill(const int batch_item_id, const DatumMap& datum_map, 
            const Dtype* mean, Dtype* filled_data, Dtype* filled_label, 
            Blob<Dtype> &depth_data); 

  void ContourFill(const int batch_item_id, const DatumMap& datum_map, 
            const Dtype* mean, Dtype* filled_data, Dtype* filled_label, 
            Blob<Dtype> &contour_data); 

 protected:
  virtual unsigned int Rand();

  // Tranformation parameters
  TransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Caffe::Phase phase_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

