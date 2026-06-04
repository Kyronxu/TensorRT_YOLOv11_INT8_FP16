#ifndef POSTPROCESS_H
#define POSTPROCESS_H


#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "common.hpp"

std::vector<det::Object> cuda_postprocess(const float* d_output, int num_classes, int num_anchors,
                    const det::PreParam& pparam, float score_threshold, float iou_threshold, int topk,
                    float* d_trans, float* d_boxes, int* d_count, int* d_keep,
                cudaStream_t stream);
__global__ void transpose_yolov11_kernel(const float* input, float* output, int channels, int anchors);



#endif
