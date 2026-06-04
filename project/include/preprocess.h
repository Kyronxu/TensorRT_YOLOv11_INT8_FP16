#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "common.hpp"

struct AffineMatrix{
    float value[6];
};

void cuda_preprocess(const uint8_t* src, int src_width, int src_height,
                     float* dst_device, int dst_width, int dst_height,
                    cudaStream_t stream, det::PreParam& pparam, uint8_t* d_src_buf);


#endif
