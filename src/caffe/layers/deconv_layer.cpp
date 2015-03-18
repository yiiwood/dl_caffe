#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DeconvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Configure the kernel size, padding, stride, and inputs.  padding param not used, by liangji,20141201 
  ConvolutionParameter deconv_param = this->layer_param_.deconvolution_param();
  CHECK(!deconv_param.has_kernel_size() !=
      !(deconv_param.has_kernel_h() && deconv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(deconv_param.has_kernel_size() ||
      (deconv_param.has_kernel_h() && deconv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!deconv_param.has_pad() && deconv_param.has_pad_h()
      && deconv_param.has_pad_w())
      || (!deconv_param.has_pad_h() && !deconv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!deconv_param.has_stride() && deconv_param.has_stride_h()
      && deconv_param.has_stride_w())
      || (!deconv_param.has_stride_h() && !deconv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  
  if (deconv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = deconv_param.kernel_size();
  } else {
    kernel_h_ = deconv_param.kernel_h();
    kernel_w_ = deconv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  
  if (!deconv_param.has_pad_h()) {
    pad_h_ = pad_w_ = deconv_param.pad();
  } else {
    pad_h_ = deconv_param.pad_h();
    pad_w_ = deconv_param.pad_w();
  }
  
  if (!deconv_param.has_stride_h()) {
    stride_h_ = stride_w_ = deconv_param.stride();
  } else {
    stride_h_ = deconv_param.stride_h();
    stride_w_ = deconv_param.stride_w();
  }
  
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();   // input channels
  num_output_ = this->layer_param_.deconvolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.deconvolution_param().group();
  CHECK_EQ(group_, 1) << "Group must set to be 1.";	
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0) << "Number of output should be multiples of group.";

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.deconvolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
	this->blobs_[0].reset(new Blob<Dtype>(
        channels_ , num_output_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.deconvolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
	
    // If necessary, initialize and fill the biases:
    // 1 x 1 x 1 x output channels
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.deconvolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }

  // Shape the tops.  upsampling
  //height_out_ = (height_ - 1) * stride_h_ + kernel_h_;
  //width_out_ = (width_ - 1) * stride_w_ + kernel_w_;
  height_out_ = height_ * stride_h_;
  width_out_ = width_ * stride_w_;
  
  M_ = channels_ ;     //filter row , input image row (how many filter numbers ,also how many input channels)
  K_ = num_output_ * kernel_h_ * kernel_w_  ; // filter col, 
  N_ = height_ * width_;                      // input image col,
  OUT_N_ = height_out_* width_out_;           //  one channel output map's number
  
  //col data of output data, shape is ( outputnum*kw*kh, h*w)
  col_buffer_.Reshape(1, num_output_ * kernel_h_ * kernel_w_, height_, width_);
	  
  for (int top_id = 0; top_id < top->size(); ++top_id) {
    (*top)[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, OUT_N_); 
    caffe_set(OUT_N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    caffe_set((*top)[i]->count(), Dtype(0), (*top)[i]->mutable_cpu_data());
	
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();

    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
        (Dtype)1., weight , bottom_data + bottom[i]->offset(n),
        (Dtype)0., col_data);
		  
      col2im_cpu(col_data, num_output_, height_out_, width_out_,
        kernel_h_, kernel_w_, pad_h_, pad_w_,
        stride_h_, stride_w_, top_data + (*top)[i]->offset(n));
			
	  if (bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_, OUT_N_, 1, 
          (Dtype)1., this->blobs_[1]->cpu_data(), bias_multiplier_.cpu_data(), 
          (Dtype)1., top_data + (*top)[i]->offset(n));
      } 
    }
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	const Dtype* weight = NULL;
	Dtype* weight_diff = NULL;
	if (this->param_propagate_down_[0]) {
		weight = this->blobs_[0]->cpu_data();
		weight_diff = this->blobs_[0]->mutable_cpu_diff();
		caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
	}
	Dtype* bias_diff = NULL;
	if (bias_term_ && this->param_propagate_down_[1]) {
		bias_diff = this->blobs_[1]->mutable_cpu_diff();
		caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
	}

	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = NULL;
        // Bias gradient, if necessary.
		if (bias_term_ && this->param_propagate_down_[1]) {
			top_diff = top[i]->cpu_diff();
			for (int n = 0; n < num_; ++n) {
				caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, OUT_N_,
					1., top_diff + top[0]->offset(n), bias_multiplier_.cpu_data(), 
					1., bias_diff);
			}
		}

		if (this->param_propagate_down_[0] || propagate_down[i]) {
			if (!top_diff) {
				top_diff = top[i]->cpu_diff();
			}
			Dtype* col_data = col_buffer_.mutable_cpu_data();
			Dtype* col_diff = col_buffer_.mutable_cpu_diff();
			const Dtype* bottom_data = (*bottom)[i]->cpu_data();
			Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();

			for (int n = 0; n < num_; ++n) {
				im2col_cpu(top_diff + top[i]->offset(n), num_output_, height_out_,
					width_out_, kernel_h_, kernel_w_, pad_h_, pad_w_,
                    stride_h_, stride_w_, col_diff);    //topdiff

                // gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {	   
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
						(Dtype)1., bottom_data + (*bottom)[i]->offset(n), col_diff, 
						(Dtype)1., weight_diff );
				}
                // gradient w.r.t. bottom data, if necessary.
				if (propagate_down[i]) {
					if (weight == NULL) {
						weight = this->blobs_[0]->cpu_data();
					}
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
						(Dtype)1., weight, col_diff,
						(Dtype)0., bottom_diff + (*bottom)[i]->offset(n));
				}
			}
		}

	} 

}

#ifdef CPU_ONLY
STUB_GPU(DeconvolutionLayer);
#endif

INSTANTIATE_CLASS(DeconvolutionLayer);

}  // namespace caffe
