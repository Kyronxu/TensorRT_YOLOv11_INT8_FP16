#include "yolov11.h"

YOLOv11::YOLOv11(const std::string& engine_file_path)
{
    // open TensorRT engine file, binary mode
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());

    //seek to the ending of file
    file.seekg(0, std::ios::end);
    auto size = file.tellg();   // size equals to file size
    file.seekg(0, std::ios::beg);

    //malloc a char array in heap to store the binary file data
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // initialize TensorRT plugin (if define a self-defined Layer)
    initLibNvInferPlugins(&this->gLogger, "");

    //create runtime obj (IRuntime), for deserializing the engine
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // deserialize the engine, make bineary flow to ICudaEngine example
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);

    //release the char array in heap
    delete[] trtModelStream;

    //base this->engine to create execution context
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    //create a CUDA Stream for Async within copy and infer
    cudaStreamCreate(&this->stream);

    //search I/O Tensor num in model
    this->num_bindings = this->engine->getNbIOTensors();

    //iter each binding, collect name, type, size and input/output
    for (int i = 0; i < this->num_bindings; i++)
    {
        det::Binding binding;
        //get binding name by index i
        std::string name = this->engine->getIOTensorName(i);
        binding.name = name;

        //get Tensor type, and calculate the size
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
        binding.dsize = type_to_size(dtype);

        //check this binding is input or output
        bool IsInput = this->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        
        //get Tensor shape, dynamic shape
        nvinfer1::Dims dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);

        if (IsInput)
        {
            //process input binding
            this->num_inputs += 1;
            binding.size = get_size_by_dim(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);

            //set max opt shape
            //info context: input Tensor's shape for dynamic infer
            this->context->setInputShape(name.c_str(), dims);
            std::cout << "input name: " << name << " dims: " << dims.nbDims
                << " input shape: [" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ", " << dims.d[3] << "]" << std::endl;
        }
        else    //process output binding
        {
            dims = this->context->getTensorShape(name.c_str());
            binding.size = get_size_by_dim(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;

            std::cout << "output name: " << name << " nbDims: " << dims.nbDims << "shape: [";
            for (int d = 0; d < dims.nbDims; ++d) std::cout << dims.d[d] << (d+1 < dims.nbDims ? ", " : "");
            std::cout << "]" << std::endl;
        }
    }
}

YOLOv11::~YOLOv11()
{
    delete this->context;
    delete this->runtime;
    delete this->engine;

    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs)
    {
        CUDA_CHECK(cudaFree(ptr));
    }
    for(auto& ptr : this->host_ptrs)
    {
        CUDA_CHECK(cudaFreeHost(ptr));
    }

    //release pre-allocated buffer
    if (d_src_buf_) {cudaFree(d_src_buf_); d_src_buf_ = nullptr;}
    if (d_trans_) {cudaFree(d_trans_); d_trans_ = nullptr;}
    if (d_boxes_) {cudaFree(d_boxes_); d_boxes_ = nullptr;}
    if (d_count_) {cudaFree(d_count_); d_count_ = nullptr;}
    if (d_keep_) {cudaFree(d_keep_); d_keep_ = nullptr;}
}

void YOLOv11::make_pipe(bool warmup)
{
    //1.Input bindings: allocate Device (GPU) memory
    for(auto& bindings : this->input_bindings)
    {
        void* d_ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    //2.output bindings: allocate Device + Host (page-locked) memory
    for (auto& bindings : this->output_bindings)
    {
        void* d_ptr = nullptr;
        void* h_ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        CUDA_CHECK(cudaHostAlloc(&h_ptr, bindings.size * bindings.dsize, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    //3. Pre-allocate preprocess buffer (up to 4K, avoids per-frame cudaMalloc)
    d_src_size_ = size_t(3840) * 2160 * 3;
    CUDA_CHECK(cudaMalloc(&d_src_buf_, d_src_size_));

    //4.Pre-allocate postprocess buffers
    max_anchors_ = static_cast<int>(this->output_bindings[0].dims.d[2]);    //e.g. 8400
    num_classes_ = static_cast<int>(this->output_bindings[0].dims.d[1]) - 4;
    CUDA_CHECK(cudaMalloc(&d_trans_, max_anchors_ * (num_classes_ + 4) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_boxes_, max_anchors_ * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keep_, max_anchors_ * sizeof(int)));

    printf("Pre-allocated: src=%.1fMB, trans=%.1fMB, boxes=%.1fMB\n",
        d_src_size_ / 1e6f, 
        max_anchors_ * (num_classes_ + 4) * sizeof(float) / 1e6f,
        max_anchors_ * 6 * sizeof(float) / 1e6f);

    //5. Optional warmup
    if (warmup)
    {
        int wdst_h = static_cast<int>(input_bindings[0].dims.d[2]);
        int wdst_w = static_cast<int>(input_bindings[0].dims.d[3]);
        size_t input_bytes = 3 * wdst_h * wdst_w * sizeof(float);

        std::vector<float> zero(input_bytes / sizeof(float), 0.0f);
        for (int i = 0; i < 10; ++i)
        {
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs[0], zero.data(), input_bytes, cudaMemcpyHostToDevice, this->stream));
            this->infer();
        }
        cudaStreamSynchronize(this->stream);
        std::cout << "[WARMUP] model warmup 10 times" << std::endl;
    }
}

void YOLOv11::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;

    float r = std::min<float>(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh)
    {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }else
    {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f; dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0,0,0), true, false, CV_32F);

    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
}

void YOLOv11::preprocessGPU(const cv::Mat& image)
{
    // 确保预分配缓冲足够大（通常 4K 已足够，此处做安全检查）
    size_t needed = size_t(image.cols) * image.rows * 3;
    if (needed > d_src_size_) {
        cudaFree(d_src_buf_);
        d_src_size_ = needed;
        CUDA_CHECK(cudaMalloc(&d_src_buf_, d_src_size_));
    }
    cuda_preprocess(image.data, image.cols, image.rows,
                    static_cast<float*>(device_ptrs[0]),
                    dst_w, dst_h, stream, pparam, d_src_buf_);
}

void YOLOv11::postprocessGPU(std::vector<det::Object>& objs, float score_thre, float iou_thre, int topk)
{
    objs.clear();
    int num_channels = this->output_bindings[0].dims.d[1];
    int num_anchors = this->output_bindings[0].dims.d[2];
    int num_classes = num_channels - 4;

    int output_start_idx = static_cast<int>(this->input_bindings.size());
    float* d_output = static_cast<float*>(this->device_ptrs[output_start_idx]);

    //use pre-allocated buffer, avoid each frame malloc/free
    objs = cuda_postprocess(
        d_output, num_classes, num_anchors,
        this->pparam, score_thre, iou_thre, topk,
        d_trans_, d_boxes_, d_count_, d_keep_,
        this->stream
    );
}

void YOLOv11::copy_from_Mat(const cv::Mat& image)
{
    //prepare a container shape of NCHW, save processed image
    cv::Mat nchw;
    
    //2.collect input bindings of first input binding from former function
    auto& in_binding = this->input_bindings[0];

    //3.get height and width of input Tensor shape
    //dims.d[3] corresponds to width, dims.d[2] corresponds to height, (NHWC->NCHW)
    auto width64 = in_binding.dims.d[3];
    auto height64 = in_binding.dims.d[2];

    //4.CUDA_CHECK
    if(width64 > INT_MAX || height64 > INT_MAX)
    {
        throw std::runtime_error("Input dimensions too large for cv::Size");
    }

    //5.convert 64 bit to int, which OpenCV requires
    cv::Size size{static_cast<int>(width64), static_cast<int>(height64)};

    //6.call letterbox, scale original image to (114,114,114)  change to NCHW Float Blob
    this->letterbox(image, nchw, size);

    //7. nchw is 1x3xhxw image with CV_32F type
    //in context, set input Tensor shape
    this->context->setInputShape(in_binding.name.c_str(), nvinfer1::Dims{4, {1, 3, height64, width64}});

    //8. let pre-allocate GPU's ptr bind to input tensor
    //enqueueV3, TensorRT known where to read input tensor
    this->context->setTensorAddress(in_binding.name.c_str(), device_ptrs[0]);

    //9.Async make data from HOST TO DEVICE
    CUDA_CHECK(cudaMemcpyAsync(
        this->device_ptrs[0],   //target:: GPU input buffer
        nchw.ptr<float>(),   //source:: nchw blob memory
        nchw.total() * nchw.elemSize(), //copy data bit: H*W*3*4
        cudaMemcpyHostToDevice,
        this->stream
    ));
}

void YOLOv11::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);

    auto& in_binding = this->input_bindings[0];

    std::string input_name = in_binding.name;
    this->context->setInputShape(input_name.c_str(), nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CUDA_CHECK(cudaMemcpyAsync(
        this->device_ptrs[0],   //target:: GPU input buffer
        nchw.ptr<float>(),   //source:: nchw blob memory
        nchw.total() * nchw.elemSize(), //copy data bit: H*W*3*4
        cudaMemcpyHostToDevice,
        this->stream
    ));
}

void YOLOv11::infer()
{
    //1.make all input and output's Tensor GPU bind to context
    /*
    * for each binding, if i < num_inputs, it is input binding, else output binding
    * call setTensorAddress to bind GPU memory, do infer in this TensorRT context
    */
   for(int i = 0; i < num_bindings; ++i)
   {
       const char* tensorname = (i < num_inputs ? input_bindings[i].name : output_bindings[i - num_inputs].name).c_str();
       void* devicePtr = device_ptrs[i];
       this->context->setTensorAddress(tensorname, devicePtr);
   }

   //2.launch async task
   // use enqueueV3 to launch async task otherwise enqueueV2/executeV2
   //implict data transfer and kernel function parrallel in same CUDA
   this->context->enqueueV3(this->stream);
   
   //3.Async, make output tensor copy to Host from GPU
   for(int i = 0; i < this->num_outputs; ++i)
   {
    size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
    CUDA_CHECK(cudaMemcpyAsync(
        this->host_ptrs[i],
        this->device_ptrs[i + this->num_inputs],
        osize,
        cudaMemcpyDeviceToHost,
        this->stream
    ));
   }
   //4.ensure copy and infer task on this Stream has been completed
   cudaStreamSynchronize(this->stream);
}

void YOLOv11::postprocess(std::vector<det::Object>& objs, float score_thres, float iou_thres, int topk)
{
    objs.clear();
    //1. get input height and width
    auto& input_h = this->input_bindings[0].dims.d[2];
    auto& input_w = this->input_bindings[0].dims.d[3];

    //2.find det output
    int num_channels = 0, num_anchors = 0, num_classes = 0;
    bool found = false;
    int bid = -1;
    for (size_t bcnt = 0; bcnt < this->output_bindings.size(); ++bcnt)
    {
        auto& o = this->output_bindings[bcnt];
        if (o.dims.nbDims == 3)
        {
            num_channels = o.dims.d[1];
            num_anchors = o.dims.d[2];
            // num_classes = num_channels - 4;
            found = true;
            bid = static_cast<int>(bcnt);
            break;
        }
    }
    assert(found);
    num_classes = num_channels - 4;

    //3.letterbox
    auto& dw = this->pparam.dw;
    auto& dh = this->pparam.dh;
    auto& width = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio = this->pparam.ratio;

    //4.warpaffine as cv::Mat
    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[bid]));
    output = output.t(); // num_anchors x num_channels

    //5.iter anchor refine box, score, label
    std::vector<int>labels;
    std::vector<float> scores;
    std::vector<cv::Rect_<float>> bboxes;
    std::vector<cv::Rect> int_bboxes;   //NMS needed

    for(int i = 0; i < num_anchors; i++)
    {
        auto row_ptr = output.row(i).ptr<float>();
        auto box_ptr = row_ptr;     //fourth is box
        auto score_ptr = row_ptr + 4;   //after is num_classes labels

        auto max_score_ptr = std::max_element(score_ptr, score_ptr + num_classes);
        float score = *max_score_ptr;

        if (score > score_thres)
        {
            float x = *box_ptr++ - dw;
            float y = *box_ptr++ - dh;
            float w = *box_ptr++;
            float h = *box_ptr;

            float x0 = std::clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = std::clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = std::clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = std::clamp((y + 0.5f * h) * ratio, 0.f, height);

            int label = max_score_ptr - score_ptr;
            cv::Rect_<float> bbox(x0, y0, x1 - x0, y1 - y0);
            bboxes.push_back(bbox);

            int_bboxes.emplace_back(cv::Rect(
                static_cast<int>(x0),
                static_cast<int>(y0),
                static_cast<int>(x1 - x0),
                static_cast<int>(y1 - y0)));
            
            labels.push_back(label);
            scores.push_back(score);
        }
    }
    // 6. NMS
    std::vector<int> indices;
#if defined(BATCHED_NMS)
    cv::dnn::NMSBoxesBatched(int_bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(int_bboxes, scores, score_thres, iou_thres, indices);
#endif

    // 7. Construct final Object, keep topk
    int cnt = 0;
    for (auto idx : indices) {
        if (cnt >= topk) break;

        det::Object obj;
        obj.rect = bboxes[idx]; // keep float precision
        obj.label = labels[idx];
        obj.prob = scores[idx];
        objs.push_back(obj);
        cnt++;
    }
}

void YOLOv11::draw_objects(
    const cv::Mat& image,                         // 输入原图
    cv::Mat& res,                                 // 输出可视化图
    const std::vector<det::Object>& objs,              // 检测结果
    const std::vector<std::string>& CLASS_NAMES,  // 类别名称
    const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs)
    {
        int idx = obj.label;
        cv::Scalar color(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);

        //2. draw rectangle
        cv::rectangle(res, obj.rect, color, 2);
        //3.draw label and text
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[idx].c_str(), obj.prob * 100);

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int x = std::max((int)obj.rect.x, 0);
        int y = std::max((int)obj.rect.y, 0);

        //4.ensure text stay inside
        if (x + label_size.width > res.cols)
            x = res.cols - label_size.width;
        if (y - label_size.height < 0)
            y = label_size.height;

        //5.draw background
        cv::rectangle(res, cv::Rect(x, y - label_size.height, label_size.width, label_size.height + baseline), color, cv::FILLED);
        cv::putText(res, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    }
}