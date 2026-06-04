#include "postprocess.h"

//decode kernel: output corner coordinate of origin image
__global__ void decode_yolov11_kernel(int num_class, int num_anchors, float conf_thresh, const float* src, float* out_boxes, int* out_count, const det::PreParam pparam)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_anchors) return;

    const float* psrc = src + idx * (num_class + 4);
    float cx = psrc[0];
    float cy = psrc[1];
    float w = psrc[2];
    float h = psrc[3];

    const float* score_ptr = psrc + 4;
    float score = score_ptr[0];
    int label = 0;
    for (int i = 1; i < num_class; i++)
    {
        if (score_ptr[i] > score)
        {
            score = score_ptr[i];
            label = i;
        }
    }
    if (score < conf_thresh) return;

    // keychange: inverse to original image coordinate
    float x0 = (cx - 0.5f*w - pparam.dw) * pparam.ratio;
    float y0 = (cy - 0.5f*h - pparam.dh) * pparam.ratio;
    float x1 = (cx + 0.5f*w - pparam.dw) * pparam.ratio;
    float y1 = (cy + 0.5f*h - pparam.dh) * pparam.ratio;

    x0 = fminf(fmaxf(x0, 0.f), pparam.width);
    y0 = fminf(fmaxf(y0, 0.f), pparam.height);
    x1 = fminf(fmaxf(x1, 0.f), pparam.width);
    y1 = fminf(fmaxf(y1, 0.f), pparam.height);

    if (x0 >= x1 || y0 >= y1) return;
    int index = atomicAdd(out_count, 1);
    if (index >= num_anchors) return;

    int base = index * 6;
    out_boxes[base + 0] = x0;
    out_boxes[base + 1] = y0;
    out_boxes[base + 2] = x1;
    out_boxes[base + 3] = y1;
    out_boxes[base + 4] = score;
    out_boxes[base + 5] = float(label);
}

// =========NMS GPU==========
__global__ void nms_kernel(const float* bboxes, int* keep_flags, int num_boxes, float iou_threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes) return;
    keep_flags[i] = 1;

    const float* box_i = bboxes + i * 6;
    for (int j = 0; j < i; j++)
    {
        if (keep_flags[j] == 0) continue;
        const float* box_j = bboxes + j * 6;

        if(fabs(box_i[5]-box_j[5]) > 1e-3f) continue;

        float left = fmaxf(box_i[0], box_j[0]);
        float top = fmaxf(box_i[1], box_j[1]);
        float right = fminf(box_i[2], box_j[2]);
        float bottom = fminf(box_i[3], box_j[3]);

        float w = fmaxf(0.f, right - left);
        float h = fmaxf(0.f, bottom - top);
        float inter = w * h;

        float area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1]);
        float area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1]);

        float iou = inter / (area_i + area_j - inter + 1e-5f);
        if (iou > iou_threshold) {keep_flags[i] = 0; return;}
    }
}

//=========transpose kernel==========
__global__ void transpose_yolov11_kernel(const float* input, float* output, int channels, int anchors)
{
    int anchor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor_idx >= anchors) return;
    for(int c = 0; c < channels; c++)
    {
        output[anchor_idx * channels + c] = input[c * anchors + anchor_idx];
    }
}

// ============GPU==========
//input: boxes [num_boxes, 6], keep_flags, num_boxes, topk
//output: topk_idx [topk]
__global__ void topk_kernel(const float* boxes, const int* keep_flags, int* topk_idx, int num_boxes, int topk)
{
    extern __shared__ float shared_probs[]; // shared memory for probabilities
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (tid < num_boxes)
    {
        shared_probs[tid] = (idx < num_boxes && keep_flags[idx]) ? boxes[idx * 6 + 4] : -1.f;
    }
    else
    {
        shared_probs[tid] = -1.f;
    }

    for (int i = 0; i < topk; i++)
    {
        float max_val = -1.f;
        int max_idx = -1;
        for (int j = 0; j < num_boxes; j++)
        {
            if (shared_probs[j] > max_val)
            {
                max_val = shared_probs[j];
                max_idx = j;
            }
        }
        if (max_idx >= 0)
        {
            if (i < topk) topk_idx[i] = max_idx;
            shared_probs[max_idx] = -1.f;   // set to -1 to avoid re-selection
        }
    }
}


//===========CUDA postprocess==========
std::vector<det::Object> cuda_postprocess(const float* d_output, int num_classes, int num_anchors,
                    const det::PreParam& pparam, float score_threshold, float iou_threshold, int topk,
                    float* d_trans, float* d_boxes, int* d_count, int* d_keep,
                cudaStream_t stream)
{
    int BLOCK = 256;
    int grid = (num_anchors + BLOCK - 1) / BLOCK;

    // transpose
    transpose_yolov11_kernel<<<grid, BLOCK, 0, stream>>>(d_output, d_trans, num_classes + 4, num_anchors);

    //reset count async
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    cudaMemsetAsync(d_boxes, 0, num_anchors * 6 * sizeof(float), stream);

    //decode (async)
    decode_yolov11_kernel<<<grid, BLOCK, 0, stream>>>(num_classes, num_anchors, score_threshold, d_trans, d_boxes, d_count, pparam);

    // sync
    cudaStreamSynchronize(stream);

    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_count == 0) return {};
    h_count = std::min(h_count, num_anchors);

    //NMS
    cudaMemsetAsync(d_keep, 0, h_count * sizeof(int), stream);
    grid = (h_count + BLOCK - 1) / BLOCK;
    nms_kernel<<<grid, BLOCK, 0, stream>>>(d_boxes, d_keep, h_count, iou_threshold);

    cudaStreamSynchronize(stream);
    std::vector<int> h_keep(h_count);
    std::vector<float> h_boxes(h_count * 6);

    cudaMemcpy(h_keep.data(), d_keep, h_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_boxes.data(), d_boxes, h_count * 6 * sizeof(float), cudaMemcpyDeviceToHost);

    //construct box
    std::vector<det::Object> objs;
    objs.reserve(topk);
    for (int i = 0; i < h_count; i++)
    {
        if (!h_keep[i]) continue;
        det::Object obj;
        obj.rect = cv::Rect_<float>(
            h_boxes[i*6+0], h_boxes[i*6+1],
            h_boxes[i*6+2] - h_boxes[i*6+0],
            h_boxes[i*6+3] - h_boxes[i*6+1]
        );
        obj.prob = h_boxes[i*6+4];
        obj.label = int(h_boxes[i*6+5]);
        objs.push_back(obj);
        if ((int)objs.size() >= topk) break;
    }
    std::sort(objs.begin(), objs.end(), [](const det::Object& a, const det::Object& b) {
        return a.prob > b.prob;
    });
    return objs;
}
