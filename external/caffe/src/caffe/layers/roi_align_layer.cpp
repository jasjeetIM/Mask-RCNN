// ------------------------------------------------------------------
// Project: Mask R-CNN 
// File: ROIAlignLayer
// Adopted from roi_pooling_layer.cpp (written by Ross Grischik)
// Author: Jasjeet Dhaliwal
// ------------------------------------------------------------------

#include <cfloat>
#include <algorithm> 
#include <stdlib.h> 
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::vector; 
using std::fabs; 
namespace caffe {

template <typename Dtype> 
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                      const vector<Blob<Dtype>*>& top) 
{
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_param.pooled_h();
  pooled_width_ = roi_align_param.pooled_w();
  spatial_scale_ = roi_align_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) 
{
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  int shape_init[] = {bottom[1]->num(), channels_, pooled_height_,
      pooled_width_, 4};
  const vector<int> shape(shape_init, shape_init + sizeof(shape_init) 
      / sizeof(int));
  max_mult_.Reshape(shape); 
  max_pts_.Reshape(shape);
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) 
{ 
  LOG(INFO) << "DOING CPU FORWARD NOW "; 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_idx = max_pts_.mutable_cpu_data();
  Dtype* argmax_mult = max_mult_.mutable_cpu_data(); 
  caffe_set(top_count*4, -1, argmax_idx);
  caffe_set(top_count*4, Dtype(-FLT_MAX), argmax_mult);
  //std::cout << "TOTAL = " << num_rois*channels_*height_*width_ << "\n"; 
  // For each ROI R = [batch_index x1 y1 x2 y2]:
  for (int n = 0; n < num_rois; ++n) {

    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale_;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale_;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale_;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale_;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);
      if (n != roi_batch_ind) {
        continue;
      }
    //Util Values
    Dtype one = 1.0; 
    Dtype zero = 0.0; 

    Dtype roi_height = max(roi_end_h - roi_start_h, one);  
    Dtype roi_width = max(roi_end_w - roi_start_w, one);
    const Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = roi_width  /  static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
    int argmax_offset_init[] = {0,1,0,0,0}; 
    const vector<int> offset_argmax(argmax_offset_init, 
                 argmax_offset_init + sizeof(argmax_offset_init) /sizeof(int)); 
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {

          Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
          Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
          Dtype hend = static_cast<Dtype>(ph + 1)* bin_size_h;
          Dtype wend =static_cast<Dtype>(pw + 1) * bin_size_w;

          hstart = min(max(hstart + roi_start_h, zero), static_cast<Dtype>(height_));
          hend = min(max(hend + roi_start_h, zero), static_cast<Dtype>(height_));
          wstart = min(max(wstart + roi_start_w, zero), static_cast<Dtype>(width_));
          wend = min(max(wend + roi_start_w, zero), static_cast<Dtype>(width_));

          Dtype maxvalue = -FLT_MAX; 
          int maxidx[4];
          Dtype maxmult[4];   
          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          int argmax_index = (ph * pooled_width_ + pw) * 4;
          if (is_empty) {
            maxvalue = 0;
            for (int i = 0; i<4; ++i) {
              maxidx[i] = -1;
              maxmult[i] = -FLT_MAX; 
            } 
          }
            Dtype samples_n[8] = {-0.5, -0.5, -0.5, 0.5, 
                                   0.5, -0.5, 0.5, 0.5}; 
            Dtype bisampled[4];
            int counter = 0;  
            Dtype x_smp_n = -2.0, y_smp_n = -2.0, h_idx_n = -2.0, w_idx_n = -2.0; 
           
            //Bilinearly Interpolate 4 sampled values
            for (int smp = 0; smp < sizeof(samples_n)/sizeof(*samples_n) ; smp+=2) {
              x_smp_n = samples_n[smp]; 
              y_smp_n = samples_n[smp+1]; 
              
              bisampled[smp/2] = 0.0;
              int b_index[4] = {-1, -1 , -1, -1}; //, -1,-1,-1,-1}; 
              int b_index_curr[4] = {-1, -1 , -1, -1}; //, -1,-1,-1,-1};
              Dtype multiplier[4] = {Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX)};   
                                       //Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX)};  

              counter = 0;
              for (int h_idx = floor(hstart); h_idx <= ceil(hend) && h_idx < height_; ++h_idx) {
                for (int w_idx = floor(wstart); w_idx <= ceil(wend) && w_idx < width_; ++w_idx) {
                  if (counter < 4) {
                    b_index[counter] = ((((n*channels_ + c) * height_) + h_idx ) * width_ )+ w_idx; 
                    b_index_curr[counter] = (h_idx*width_) + w_idx; 
                    //Normalize h_idx and w_idx
                    h_idx_n =  static_cast<Dtype>( (2*(static_cast<Dtype>(h_idx) - roi_start_h) / (roi_end_h - roi_start_h)) - 1);
                    w_idx_n =  static_cast<Dtype>( (2*(static_cast<Dtype>(w_idx) - roi_start_w) / (roi_end_w - roi_start_w))  - 1);
                    h_idx_n = min(max(h_idx_n, static_cast<Dtype>(-1.0)),one);
                    w_idx_n = min(max(w_idx_n, static_cast<Dtype>(-1.0)),one);
                    multiplier[counter] = max(zero,static_cast<Dtype>(1 - fabs(x_smp_n - w_idx_n))) 
                                             * max(zero,static_cast<Dtype>(1 - fabs(y_smp_n - h_idx_n)));

                    bisampled[smp/2] += batch_data[b_index_curr[counter]]*multiplier[counter];  
                    ++counter; 
		 } else { 
		    goto stop;  
		 }
                } // w_idx
              } //h_idx
              stop:
              if (bisampled[smp/2] > maxvalue) {
                maxvalue = bisampled[smp/2];
                for (int i=0; i<4;++i) {
                  maxidx[i] = b_index[i];
                  maxmult[i] = multiplier[i];  
	        }

              }
            } //smp
            //Store value in the top blob
            top_data[pool_index] = maxvalue;
            for (int i = 0; i<4; ++i, ++argmax_index) {
              argmax_idx[argmax_index] = maxidx[i]; 
              argmax_mult[argmax_index] = maxmult[i]; 
            }
        } //pw
      } // ph
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      if ( (c+1) < channels_ ){
        argmax_idx += max_pts_.offset(offset_argmax);
        argmax_mult += max_mult_.offset(offset_argmax); 
      }
    } // channels
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }//num_rois
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* bottom_rois = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int count = bottom[0]->count();
  caffe_set(count, Dtype(0.), bottom_diff);
  
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num(); 
  const int* argmax_idx = max_pts_.cpu_data();
  const Dtype* argmax_mult = max_mult_.cpu_data(); 
  
  int index = 0; //Current index
 // std::cout <<"Batch = " << batch_size << "\n";
  for (int b = 0; b < batch_size; ++b){ 
    for (int c = 0; c < channels_; ++c){
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          index = ( ( ( ( (b * channels_ ) + c ) * height_ ) + h) * width_) + w; 
  	  // Go over every ROI 
          Dtype gradient = 0.0; 
  	  for (int n = 0; n < num_rois; ++n) {
    	    const Dtype* offset_bottom_rois = bottom_rois + n * 5;
            int roi_batch_ind = offset_bottom_rois[0];
            CHECK_GE(roi_batch_ind, 0);
    	    CHECK_LT(roi_batch_ind, batch_size);

            int offset = (n * channels_ + c) * pooled_height_ * pooled_width_;
            int argmax_offset = offset * 4;
            const Dtype* offset_top_diff = top_diff + offset;
            const int* offset_argmax_idx = argmax_idx + argmax_offset;
            const Dtype* offset_argmax_mult = argmax_mult + argmax_offset;
            Dtype multiplier = 0.0;  
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                for (int k = 0; k < 4; ++k) {
                  if (offset_argmax_idx[((ph * pooled_width_ + pw) * 4) + k] == index) {
                    multiplier = offset_argmax_mult[( (ph * pooled_width_ + pw) * 4) + k];
                    gradient+= offset_top_diff[ph * pooled_width_ + pw] * multiplier; 
                  }
                } 
              }//Pw
            } //Ph 
	}// rois
        bottom_diff[index] = gradient; 
      }// width
    }//height
   }//channels
  }//count
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
