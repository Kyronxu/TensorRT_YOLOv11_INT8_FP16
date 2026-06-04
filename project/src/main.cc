#include "yolov11.h"
#include <opencv2/opencv.hpp>
#include <chrono>


int main(int argc, char** argv)
{
    cudaSetDevice(0);

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " tensorrt_engine.trt" << std::endl;
        return -1;
    }
    const std::string engine_file_path{argv[1]};
    int topk = 100;
    float score_thres = 0.25f;
    float iou_thres = 0.65f;
    std::vector<det::Object> objs;
    cv::Mat image, res;

    auto yolov11 = new YOLOv11(engine_file_path);
    yolov11->make_pipe(true);
    //CUDA Event for timing, more precise than std::chrono
    cudaEvent_t ev_start, ev_infer_end, ev_total_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_infer_end);
    cudaEventCreate(&ev_total_end);

    bool firstFrameSaved = false;


    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }
    cv::namedWindow("YOLOv11 Detection", cv::WINDOW_AUTOSIZE);
    while(true)
    {
        cap >> image;
        if (image.empty())
        {
            std::cerr << "Failed to read image" << std::endl;
            break;
        }
        if (!firstFrameSaved)
        {
            cv::imwrite("/home/xcmg/CmakeWork/SLAM_demo/SmartVehicle/TensorRT_INT8/project/build/frame.jpg", image);
            firstFrameSaved = true;
        }

        // ====== Timing ======
        cudaEventRecord(ev_start, yolov11->getStream());
        yolov11->preprocessGPU(image);
        yolov11->infer();
        cudaEventRecord(ev_infer_end, yolov11->getStream());
        cudaEventSynchronize(ev_infer_end);

        objs.clear();
        yolov11->postprocessGPU(objs, score_thres, iou_thres, topk);
        cudaEventRecord(ev_total_end, yolov11->getStream());
        cudaEventSynchronize(ev_total_end);

        // ====== end of timing ======
        float ms_infer = 0, ms_total = 0;
        cudaEventElapsedTime(&ms_infer, ev_start, ev_infer_end);
        cudaEventElapsedTime(&ms_total, ev_start, ev_total_end);

        printf("preprocess and infer: %.2f ms | postprocess: %.2f ms | total: %.2f ms | FPS: %.1f\n", ms_infer, ms_total - ms_infer, ms_total, 1000.f / ms_total);
        
        res = image.clone();
        if (!objs.empty())
        {
            yolov11->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
        }
        cv::imshow("result", res);
        if (cv::waitKey(1) == 'q') break;
    }
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_infer_end);
    cudaEventDestroy(ev_total_end);

    cap.release();
    cv::destroyAllWindows();
    delete yolov11;
    return 0;

}