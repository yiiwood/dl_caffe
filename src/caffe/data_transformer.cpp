#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Jitter(DatumMap& datum_map,
  const bool mirror, const float scale_range, const float rotate_range) {

  // datum size
  int datum_channels = datum_map.datum_channels();
  int datum_height = datum_map.datum_height();
  int datum_width = datum_map.datum_width();
  CHECK_EQ( datum_channels, 3);

  // label size
  int label_channels = datum_map.label_channels();
  int label_height = datum_map.label_height();
  int label_width = datum_map.label_width();
  CHECK_EQ( label_channels, 1);

  const string& data = datum_map.data();
  cv::Mat data_mat = cv::Mat::zeros(datum_height, datum_width, CV_32FC3);
  if (data.size()) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < data_mat.rows; ++h) {
        for (int w = 0; w < data_mat.cols; ++w) {
          int idx = (c * data_mat.rows + h) * data_mat.cols + w;
          float datum_element = static_cast<float>(static_cast<uint8_t>(data[idx]));
          data_mat.at<cv::Vec3f>(h, w)[c] = datum_element;
        }   
      }   
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < data_mat.rows; ++h) {
        for (int w = 0; w < data_mat.cols; ++w) {
          int idx = (c * data_mat.rows + h) * data_mat.cols + w;
          data_mat.at<cv::Vec3f>(h, w)[c] = datum_map.float_data(idx);
        }   
      }   
    }
  }

  //int valid_num = 0;
  cv::Mat label_mat = cv::Mat::zeros(label_height, label_width, CV_32SC1);
  for (int h = 0; h < label_mat.rows; ++h) {
    for (int w = 0; w < label_mat.cols; ++w) {
      int idx = h * label_mat.cols + w;
      label_mat.at<int>(h, w) = static_cast<int>(datum_map.label_map(idx) );
      //if (label_mat.at<int>(h, w) != -1) ++valid_num;
    }   
  }
  //LOG(INFO) << "before jitter: " << valid_num;

  // jitter
  // mirror
  if (mirror && Rand() % 2) {
    cv::Mat data_mirror, label_mirror;
    cv::flip(data_mat, data_mirror, 1);            // horizontal
    cv::flip(label_mat, label_mirror, 1);
    data_mirror.copyTo( data_mat );
    label_mirror.copyTo( label_mat );
    //LOG(INFO) << "mirror jitter";
  }

  // scale
  if (scale_range > 0) {
    float scale_rate = ((Rand() % INT_MAX)*1.0 / (INT_MAX-1)) * scale_range;
    //float scale_rate = 0.25;
    int data_dh = static_cast<int>(scale_rate * datum_height) 
                - static_cast<int>(scale_rate * datum_height) % 4;
    int data_dw = static_cast<int>(scale_rate * datum_width) 
                - static_cast<int>(scale_rate * datum_width) % 4;
    int label_dh = data_dh / 4;
    int label_dw = data_dw / 4;
    //LOG(INFO) << data_dh << " "<<data_dw << " "<<label_dh << " " << label_dw; 

    if (Rand() % 2) {              // scale down
      //LOG(INFO) <<  "scale down";
      cv::Mat data_scale = cv::Mat::ones(datum_height + 2*data_dh, datum_width + 2*data_dw, CV_32FC3) * 127.5;
      data_mat.copyTo( data_scale(cv::Range(data_dh, datum_height + data_dh), cv::Range(data_dw, datum_width + data_dw)) );
      cv::resize(data_scale, data_mat, cv::Size(datum_height, datum_width));

      cv::Mat label_scale = cv::Mat::ones(label_height + 2*label_dh, label_width + 2*label_dw, CV_32SC1) * -1;
      label_mat.copyTo( label_scale(cv::Range(label_dh, label_height + label_dh), cv::Range(label_dw, label_width + label_dw)) );
      cv::resize(label_scale, label_mat, cv::Size(label_height, label_width), 0, 0, CV_INTER_NN);
    }
    else {                         // scale up
      //LOG(INFO) <<  "scale up";
      cv::Mat data_scale;
      cv::resize(data_mat(cv::Range(data_dh, datum_height - data_dh), cv::Range(data_dw, datum_width - data_dw)), 
        data_scale, cv::Size(datum_height, datum_width));
      data_scale.copyTo( data_mat );

      cv::Mat label_scale;
      cv::resize(label_mat(cv::Range(label_dh, label_height - label_dh), cv::Range(label_dw, label_width - label_dw)), 
        label_scale, cv::Size(label_height, label_width), 0, 0, CV_INTER_NN);
      label_scale.copyTo( label_mat );
    }
    //LOG(INFO) << "scale jitter";
  }

  // rotate
  if (rotate_range > 0) {
    float rotate_rate = ((Rand() % INT_MAX)*1.0 / (INT_MAX-1)) * rotate_range;
    //float rotate_rate = 90;
    if (Rand() % 2) {   // 顺时针
      rotate_rate *= -1;
    }
    //LOG(INFO) << rotate_rate; 
    cv::Mat data_rotate, label_rotate;

    cv::Mat d_rot_mat = cv::getRotationMatrix2D( cv::Point2f(datum_width/2., datum_height/2.), rotate_rate, 1 );
    //LOG(INFO) << d_rot_mat.at<double>(0, 0) << " "<< d_rot_mat.at<double>(0, 1) << " "<< d_rot_mat.at<double>(0, 2);
    //LOG(INFO) << d_rot_mat.at<double>(1, 0) << " "<< d_rot_mat.at<double>(1, 1) << " "<< d_rot_mat.at<double>(1, 2);
    cv::warpAffine(data_mat, data_rotate, d_rot_mat, data_mat.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(127.5));

    cv::Mat l_rot_mat = cv::getRotationMatrix2D( cv::Point2f(label_width/2., label_height/2.), rotate_rate, 1 );
    //LOG(INFO) << l_rot_mat.at<double>(0, 0) << " "<< l_rot_mat.at<double>(0, 1) << " "<< l_rot_mat.at<double>(0, 2);
    //LOG(INFO) << l_rot_mat.at<double>(1, 0) << " "<< l_rot_mat.at<double>(1, 1) << " "<< l_rot_mat.at<double>(1, 2);
    cv::warpAffine(label_mat, label_rotate, l_rot_mat, label_mat.size(), CV_INTER_NN, cv::BORDER_CONSTANT, cv::Scalar(-1));
    data_rotate.copyTo( data_mat );
    label_rotate.copyTo( label_mat );
    //LOG(INFO) << "rotate jitter";
  }

  // save to datum_map 
  datum_map.clear_data();
  datum_map.clear_float_data();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < data_mat.rows; ++h) {
      for (int w = 0; w < data_mat.cols; ++w) {
        datum_map.add_float_data( data_mat.at<cv::Vec3f>(h, w)[c] );
      }   
    }   
  }
  //int valid_num_jitter = 0;
  datum_map.clear_label_map();
  for (int h = 0; h < label_mat.rows; ++h) {
    for (int w = 0; w < label_mat.cols; ++w) {
      datum_map.add_label_map( label_mat.at<int>(h, w) );
      //if ( int(datum_map.label_map(h*label_mat.cols + w)) != -1) ++valid_num_jitter;
    }
  }
  //LOG(INFO) << " after jitter: " << valid_num_jitter;
}

template<typename Dtype>
void DataTransformer<Dtype>::FillData(const int batch_item_id, 
                                      const DatumMap& datum_map, const Dtype* mean, 
                                      const int height, const int width, Dtype* filled_data) {
  // datum size
  int datum_channels = datum_map.datum_channels();
  int datum_height = datum_map.datum_height();
  int datum_width = datum_map.datum_width();
  CHECK_EQ( datum_channels, 3);
  int datum_size = datum_map.datum_channels() * height * width;
  const Dtype scale = param_.scale();

  if (height != datum_height || width != datum_width) {
    cv::Mat data_mat;
    DatumMapToDataMat(datum_map, mean, data_mat);
    cv::Mat rs_data;
    cv::resize(data_mat, rs_data, cv::Size(height, width)); 
    //FillFromMat();
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = (c * height + h) * width + w;
          filled_data[batch_item_id * datum_size + idx] = rs_data.at<cv::Vec3f>(h, w)[c];
        }
      }
    }
  }
  else {
    //FillFromDatumMap();
    const string& data = datum_map.data();
    if (data.size()) {
      for (int j = 0; j < datum_size; ++j) {
        Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        filled_data[batch_item_id * datum_size + j] = (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < datum_size; ++j) {
        filled_data[batch_item_id * datum_size + j] = (datum_map.float_data(j) - mean[j]) * scale;
      }
    }
  }

}
template<typename Dtype>
void DataTransformer<Dtype>::FillLabel(const int batch_item_id, const DatumMap& datum_map,
                                       const int height, const int width, Dtype* filled_label) {
  int label_channels = datum_map.label_channels();
  int label_height = datum_map.label_height();
  int label_width = datum_map.label_width();
  CHECK_EQ( label_channels, 1);
  int label_size = datum_map.label_channels() * height * width;

  if (height != label_height || width != label_width) {
    cv::Mat label_mat;
    DatumMapToLabelMat( datum_map, label_mat);
    cv::Mat rs_label;
    cv::resize(label_mat, rs_label, cv::Size(height, width), 0, 0, CV_INTER_NN);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int idx = h * width + w;
        filled_label[batch_item_id * label_size + idx] = rs_label.at<int>(h, w);
      }
    }
  } else {
    for (int j = 0; j < label_size; ++j) {
      filled_label[batch_item_id * label_size + j] = datum_map.label_map(j);
    }
  }

}
template<typename Dtype>
void DataTransformer<Dtype>::DatumMapToDataMat(const DatumMap& datum_map, const Dtype* mean, cv::Mat &data_mat) {
  int datum_channels = datum_map.datum_channels();
  int datum_height = datum_map.datum_height();
  int datum_width = datum_map.datum_width();
  CHECK_EQ( datum_channels, 3);
  const Dtype scale = param_.scale();
  
  data_mat.create(datum_height, datum_width, CV_32FC3);
  const string& data = datum_map.data();
  if (data.size()) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < data_mat.rows; ++h) {
        for (int w = 0; w < data_mat.cols; ++w) {
          int idx = (c * data_mat.rows + h) * data_mat.cols + w;
          float datum_element = static_cast<float>(static_cast<uint8_t>(data[idx]));
          data_mat.at<cv::Vec3f>(h, w)[c] = (datum_element - mean[idx]) * scale;
        }
      }
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < data_mat.rows; ++h) {
        for (int w = 0; w < data_mat.cols; ++w) {
          int idx = (c * data_mat.rows + h) * data_mat.cols + w;
          data_mat.at<cv::Vec3f>(h, w)[c] = (datum_map.float_data(idx) - mean[idx]) * scale;
        }
      }
    }
  }

}
template<typename Dtype>
void DataTransformer<Dtype>::DatumMapToLabelMat(const DatumMap& datum_map, cv::Mat &label_mat) {
  int label_channels = datum_map.label_channels();
  int label_height = datum_map.label_height();
  int label_width = datum_map.label_width();
  CHECK_EQ( label_channels, 1);

  label_mat.create(label_height, label_width, CV_32SC1);
  for (int h = 0; h < label_mat.rows; ++h) {
    for (int w = 0; w < label_mat.cols; ++w) {
      int idx = h * label_mat.cols + w;
      label_mat.at<int>(h, w) = static_cast<int>(datum_map.label_map(idx) );
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Fill(const int batch_item_id,
                                  const DatumMap& datum_map,
                                  const Dtype* mean,
                                  Dtype* filled_data, Dtype* filled_label) {
  const string& data = datum_map.data();
  const int datum_size = datum_map.datum_channels() * datum_map.datum_height() * datum_map.datum_width();
  const int label_size = datum_map.label_channels() * datum_map.label_height() * datum_map.label_width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
  
  if (mirror || (crop_size != 0) ) {
    LOG(FATAL) << "Current implementation does not support mirror and crop in DataTransformer::Fill!";
  }

  // we will prefer to use data() first, and then try float_data()
  if (data.size()) {
    for (int j = 0; j < datum_size; ++j) {
      Dtype datum_element =
      static_cast<Dtype>(static_cast<uint8_t>(data[j]));
      filled_data[j + batch_item_id * datum_size] =
      (datum_element - mean[j]) * scale;
    }
  } else {
    for (int j = 0; j < datum_size; ++j) {
      filled_data[j + batch_item_id * datum_size] =
      (datum_map.float_data(j) - mean[j]) * scale;
    }
  }

  if ( filled_label ) {
    for (int j = 0; j < label_size; ++j) {
      filled_label[batch_item_id * label_size + j] = datum_map.label_map(j);
    }   
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Fill(const int batch_item_id,
                                  const Datum& datum,
                                  const Dtype* mean,
                                  Dtype* filled_data) {
  const int datum_size = datum.channels() * datum.height() * datum.width();
  const Dtype scale = param_.scale();

  for (int j = 0; j < datum_size; ++j) {
    filled_data[j + batch_item_id * datum_size] =
    (datum.float_data(j) - mean[j]) * scale;
    if(isnan(filled_data[j + batch_item_id * datum_size]) 
       || isinf(filled_data[j + batch_item_id * datum_size])) {
      LOG(ERROR) << "ERROR: " << filled_data[j + batch_item_id * datum_size] 
                              << " " << datum.float_data(j) << " " << mean[j];
    }
  }

}


template<typename Dtype>
void DataTransformer<Dtype>::Pyramid(const int batch_item_id,
                                  const DatumMap& datum_map,
                                  const Dtype* mean,
                                  vector<Blob<Dtype>* > &pyramid_data, const int level_num) {

  const string& data = datum_map.data();
  const Dtype scale = param_.scale();
  int channels = datum_map.datum_channels();
  if (channels != 3) {
      LOG(FATAL) << "The channels must be 3!";
  }

  cv::Mat data_mat = cv::Mat::zeros(datum_map.datum_height(), datum_map.datum_width(), CV_32FC3);
  if (data.size()) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < data_mat.rows; ++h) {
        for (int w = 0; w < data_mat.cols; ++w) {
          int idx = (c * data_mat.rows + h) * data_mat.cols + w;
          Dtype datum_element = static_cast<float>(static_cast<uint8_t>(data[idx]));
          data_mat.at<cv::Vec3f>(h, w)[c] = (datum_element - mean[idx]) * scale;
        }   
      }   
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < data_mat.rows; ++h) {
        for (int w = 0; w < data_mat.cols; ++w) {
          int idx = (c * data_mat.rows + h) * data_mat.cols + w;
          data_mat.at<cv::Vec3f>(h, w)[c] = (datum_map.float_data(idx) - mean[idx]) * scale;
        }   
      }   
    }
  }

  for (int i = 1; i <= level_num; ++i) {
    int pyramid_channels = pyramid_data[i-1]->channels();
    int pyramid_height = pyramid_data[i-1]->height();
    int pyramid_width = pyramid_data[i-1]->width();
    int pyramid_size = pyramid_channels * pyramid_height * pyramid_width;
    Dtype* pyr_data = pyramid_data[i-1]->mutable_cpu_data();
    if (pyramid_channels != 3) {
      LOG(FATAL) << "The pyramid_channels must be 3!";
    }
    
    cv::Mat pyramid_mat = cv::Mat::zeros(pyramid_height, pyramid_width, CV_32FC3);
    // pyramid and downsampling
    cv::pyrDown(data_mat, pyramid_mat, cv::Size(pyramid_width, pyramid_height) );
    // copy to 
    pyramid_mat.copyTo( data_mat );

    // fill
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < pyramid_mat.rows; ++h) {
        for (int w = 0; w < pyramid_mat.cols; ++w) {
          int idx = (c * pyramid_mat.rows + h) * pyramid_mat.cols + w;
          pyr_data[batch_item_id * pyramid_size + idx] = pyramid_mat.at<cv::Vec3f>(h, w)[c];
        }   
      }   
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::DepthFill(const int batch_item_id, 
                                      const DatumMap& datum_map, 
                                      const Dtype* mean, 
                                      Dtype* filled_data, Dtype* filled_label,
                                      Blob<Dtype> &depth_data) {
  const string& data = datum_map.data();
  const int datum_size = datum_map.datum_channels() * datum_map.datum_height() * datum_map.datum_width();
  const int label_size = 1 * datum_map.label_height() * datum_map.label_width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
  
  if (mirror || (crop_size != 0) ) {
    LOG(FATAL) << "Current implementation does not support mirror and crop in DataTransformer::Fill!";
  }

  // we will prefer to use data() first, and then try float_data()
  if (data.size()) {
    for (int j = 0; j < datum_size; ++j) {
      Dtype datum_element =
      static_cast<Dtype>(static_cast<uint8_t>(data[j]));
      filled_data[j + batch_item_id * datum_size] =
      (datum_element - mean[j]) * scale;
    }
  } else {
    for (int j = 0; j < datum_size; ++j) {
      filled_data[j + batch_item_id * datum_size] =
      (datum_map.float_data(j) - mean[j]) * scale;
    }
  }

  if ( filled_label ) {
    for (int j = 0; j < label_size; ++j) {
      filled_label[batch_item_id * label_size + j] = datum_map.label_map(j);
    }   

    Dtype* d_data = depth_data.mutable_cpu_data();
    for (int j = 0; j < label_size; ++j) {
      int depth = int( datum_map.label_map(label_size+j) );
      d_data[batch_item_id * label_size + j] = depth;
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::ContourFill(const int batch_item_id, 
                                      const DatumMap& datum_map, 
                                      const Dtype* mean, 
                                      Dtype* filled_data, Dtype* filled_label,
                                      Blob<Dtype> &contour_data) {
  const string& data = datum_map.data();
  const int datum_size = datum_map.datum_channels() * datum_map.datum_height() * datum_map.datum_width();
  const int label_size = datum_map.label_channels() * datum_map.label_height() * datum_map.label_width();
  const int label_height = datum_map.label_height();
  const int label_width  = datum_map.label_width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
  
  if (mirror || (crop_size != 0) ) {
    LOG(FATAL) << "Current implementation does not support mirror and crop in DataTransformer::Fill!";
  }

  // we will prefer to use data() first, and then try float_data()
  if (data.size()) {
    for (int j = 0; j < datum_size; ++j) {
      Dtype datum_element =
      static_cast<Dtype>(static_cast<uint8_t>(data[j]));
      filled_data[j + batch_item_id * datum_size] =
      (datum_element - mean[j]) * scale;
    }
  } else {
    for (int j = 0; j < datum_size; ++j) {
      filled_data[j + batch_item_id * datum_size] =
      (datum_map.float_data(j) - mean[j]) * scale;
    }
  }

  if ( filled_label ) {
    for (int j = 0; j < label_size; ++j) {
      filled_label[batch_item_id * label_size + j] = datum_map.label_map(j);
    }   

    Dtype* ct_data = contour_data.mutable_cpu_data();
    int neighbour4h[4] = {0,  1,  0, -1};
    int neighbour4w[4] = {1,  0, -1,  0};
    for (int h = 0; h < label_height; ++h) {
      for (int w = 0; w < label_width; ++w) {
        int idx = h * label_width + w;
        int label = int( datum_map.label_map(idx) );
        ct_data[batch_item_id * label_size + idx] = 0;

        if (label < 0) {
          ct_data[batch_item_id * label_size + idx] = label;
        } else {
          for (int n = 0; n < 4; ++n) {
            int hh = h + neighbour4h[n];
            int ww = w + neighbour4w[n];
            if (hh >= 0 && hh < label_height && ww >=0 && ww < label_width) {
              int nlabel = int( datum_map.label_map(hh * label_width + ww) );
              if (nlabel != label) {
                ct_data[batch_item_id * label_size + idx] = 1;
                break;
              }
            }
          }
        }

      }
    }
  }
  
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  //const bool needs_rand = (phase_ == Caffe::TRAIN) &&
  //    (param_.mirror() || param_.crop_size());
  const bool needs_rand = true;
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
