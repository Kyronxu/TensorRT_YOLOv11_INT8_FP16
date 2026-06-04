#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <cuda_runtime_api.h>
#include <cuda.h>

#ifndef CHECK
#define CHECK(callstr) \
    { \
        auto status = callstr; \
        if (status != cudaSuccess) \
        { \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            assert(0); \
        } \
    }
#endif  // CHECK

#endif  // CUDA_UTILS_H
