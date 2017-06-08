// ------------------------------------------------------------------
// Project: Mask R-CNN 
// File: ROIAlignLayer
// Adopted from roi_pooling_layer.cu (written by Ross Grischik)
// Author: Jasjeet Dhaliwal
// ------------------------------------------------------------------

#include <cfloat>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <algorithm> 
#include <stdlib.h> 

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor; 
using std::ceil; 
using std::fabs; 
using std::cout; 

namespace caffe {

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_idx, Dtype* argmax_mult) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int argmax_index = index * 4; 

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;
    
    //Util Values
    Dtype zero = 0.0, one = 1.0;
 
    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w + 1.0, one);
    Dtype roi_height = max(roi_end_h - roi_start_h + 1.0, one);
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
    Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
    Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h;
    Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w;

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, zero), static_cast<Dtype>(height) );
    hend = min(max(hend + roi_start_h, zero), static_cast<Dtype>(height));
    wstart = min(max(wstart + roi_start_w, zero), static_cast<Dtype>(width));
    wend = min(max(wend + roi_start_w, zero), static_cast<Dtype>(width));
    
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero

   

    Dtype maxvalue = is_empty ? 0 : -FLT_MAX;
    int maxidx[4];
    Dtype maxmult[4];  
    //int bottom_offset =  (roi_batch_ind * channels + c) * height * width ;
    //bottom_data += (roi_batch_ind * channels + c) * height * width;
    /* Normalization function - normalizes values between -1 and 1. 
    a = -1, b = 1
    y = f(x) = [[(b - a) (x - roi_start_h)] / [roi_end_h - roi_start_h]] + a
    x = f^{-1}(y) = [[(f(x) - a)(roi_end_h - roi_end_h)] / (b - a)] + roi_start_h 
    Normalized coordinates of 4 regularly sampled points in the ROI: 
    sn_1 = (-0.5,-0.5) 
    sn_2 = (-0.5,0.5) 
    sn_3 = (0.5,-0.5) 
    sn_4 = (0.5,0.5)

    // Debugging purposes
    Dtype x_pos = (((0.5 + 1)*(roi_end_w - roi_start_w))/2.0) + roi_start_w;
    Dtype x_neg = (((-0.5 + 1)*(roi_end_w - roi_start_w))/2.0) + roi_start_w;
    Dtype y_pos = (((0.5 + 1)*(roi_end_h - roi_start_h))/2.0) + roi_start_h;
    Dtype y_neg = (((-0.5 + 1)*(roi_end_h - roi_start_h))/2.0) + roi_start_h;
    Dtype samples[2] = {x_neg, y_neg, x_neg, y_pos,
                        x_pos, y_neg, x_pos, y_pos};
    */
    
    Dtype samples_n[8] = {-0.5, -0.5, -0.5, 0.5, 
                           0.5, -0.5, 0.5, 0.5}; 
    //Holds interpolated values for each sample point
    Dtype bisampled[4]; 
    int counter = 0; 
    Dtype x_smp_n = -2.0, y_smp_n = -2.0, h_idx_n = -2.0, w_idx_n = -2.0;   
    //Bilinearly Interpolate 4 sampled values
    for (int smp = 0; smp < sizeof(samples_n)/sizeof(*samples_n) ; smp+=2) {
      x_smp_n = samples_n[smp]; 
      y_smp_n = samples_n[smp+1]; 

      bisampled[smp/2] = 0.0;
      int b_index[4] = {-1, -1 , -1, -1}; // -1,-1,-1,-1}; 
      //int b_index_curr[4] = {-1,-1,-1,-1}; 
      Dtype multiplier[4] = {Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX)}; 
					//Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX), Dtype(-FLT_MAX)};  
      counter = 0; 
      //ceil(hstart)
      //floor(hend)
      for (int h_idx = ceil(hstart); h_idx <= floor(hend) && h_idx <= height && h_idx >= 0 ; ++h_idx) {
        for (int w_idx =ceil(wstart); w_idx <= floor(wend) && w_idx <= width && w_idx >= 0; ++w_idx) {
       if (counter < 4) {
            b_index[counter] =  ((((roi_batch_ind * channels) + c) * height) + h_idx) * width + w_idx; 
        //    b_index_curr[counter]= h_idx*width + w_idx; 
            //Normalize width and height to lie between -1 and 1
            h_idx_n = static_cast<Dtype>( (static_cast<Dtype>(2)*(static_cast<Dtype>(h_idx) - roi_start_h) / (roi_end_h - roi_start_h)) - 1); 
            w_idx_n =  static_cast<Dtype>((static_cast<Dtype>(2)*(static_cast<Dtype>(w_idx) - roi_start_w) / (roi_end_w - roi_start_w))  - 1);
            h_idx_n = min(max(h_idx_n, static_cast<Dtype>(-1.0)),one);  
            w_idx_n = min(max(w_idx_n, static_cast<Dtype>(-1.0)),one);  
            multiplier[counter]=  max(zero ,static_cast<Dtype>(1 - fabs(x_smp_n - w_idx_n))) * max(zero,static_cast<Dtype>(1 - fabs(y_smp_n - h_idx_n)));
            //bisampled[smp/2] += multiplier[counter]; 
            bisampled[smp/2] += bottom_data[ b_index[counter]] * multiplier[counter];
            ++counter; 
	  } else {
	     goto stop; 
	  }
        } //w
      }//h
      stop:
      if (bisampled[smp/2] > maxvalue) {
        maxvalue = bisampled[smp/2]; 
        //Using two loops to comply with c++ convention
        for (int i=0; i<4;++i) {
          maxidx[i] = b_index[i];
          maxmult[i] = multiplier[i]; 
	}

      }
    } //smp
    //Store value in the top blob
    top_data[index] = maxvalue;
    for (int i = 0; i<4; ++i, ++argmax_index) {
      argmax_idx[argmax_index] = maxidx[i];
      argmax_mult[argmax_index] = maxmult[i]; 
    }
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_idx = max_pts_.mutable_gpu_data();
  Dtype* argmax_mult = max_mult_.mutable_gpu_data(); 
  int count = top[0]->count();
  LOG(INFO) << "Doing forward now";   
  // NOLINT_NEXT_LINE(whitespace/operators)
  //Change CAFFE_CUDA_NUM_THREADS to 64
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), 32>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_idx, argmax_mult);
   LOG(INFO) << "Done forward "; 
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_idx, const Dtype* argmax_mult, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    Dtype gradient = 0.0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      //const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      //int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
     // if (n != roi_batch_ind) {
       // continue;
     // }
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = ceil(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = ceil(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = floor(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = floor(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      int argmax_offset = offset * 4; 
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_idx = argmax_idx + argmax_offset;
      const Dtype* offset_argmax_mult = argmax_mult + argmax_offset;
      // Util Vals
      Dtype multiplier = 0.0; 
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          for (int k = 0; k < 4; ++k) {
            if (offset_argmax_idx[((ph * pooled_width + pw) * 4) + k] == index  ) {
              multiplier = offset_argmax_mult[( (ph * pooled_width + pw) * 4) + k]; 
              gradient += offset_top_diff[ph * pooled_width + pw] * multiplier;
            }
	  }
        }//pw
      }//ph
    }//rois
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_idx = max_pts_.gpu_data();
  const Dtype* argmax_mult = max_mult_.gpu_data(); 
  // NOLINT_NEXT_LINE(whitespace/operators)
  // CAFFE_CUDA_NUM_THREADS replaced with 64
   LOG(INFO) << "Doing backward "; 
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), 16>>>(
      count, top_diff, argmax_idx, argmax_mult, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  LOG(INFO) << "Done backward"; 
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
