#include "int8_entropy_calibrator.h"

#define DYNAMIC_BATCH

static constexpr int DYN_BATCH_MIN = 1;
static constexpr int DYN_BATCH_OPT = 6;
static constexpr int DYN_BATCH_MAX = 12;

static constexpr int STATIC_BATCH = 1;


int main()
{
    Logger logger;
    const char* calibrationImagesDir = "../data/";
    const char* cacheFile = "../calibration_data.cache";
    // const char* pathToOnnx = "../onnx_model/yolo11s_dynamic_batch.onnx";

    #ifdef DYNAMIC_BATCH
        const char* pathToOnnx = "../onnx_model/yolo11s_dynamic_batch.onnx";
        const char* pathToEngine = "../weights/yolo11s_int8_dynamic.engine";
    #else
        const char* pathToOnnx = "../onnx_model/yolo11s.onnx";
        const char* pathToEngine = "../weights/yolo11s_int8_static.engine";
    #endif

    /*
    * ======= Create Builder / Network ======
    */
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    initLibNvInferPlugins(&logger, "");

    //create network
    uint32_t flag = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);


    /*
    *=======parser ONNX file ======
    */
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(pathToOnnx, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));

    //ONNX's H/W (if STATIC pitch else DYNAMIC -1)
    auto inputDims = network->getInput(0)->getDimensions();
    int onnx_H = static_cast<int>(inputDims.d[2]);
    int onnx_W = static_cast<int>(inputDims.d[3]);

    //dynamic batch
    int infer_H = (onnx_H > 0) ? onnx_H : 640;
    int infer_W = (onnx_W > 0) ? onnx_W : 640;
    std::vector<int> sizeList = getTensorSizes(network);

    /*
    *=======Optimization Profile ======
    */
   nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

   #ifdef DYNAMIC_BATCH
        profile->setDimensions(
            network->getInput(0)->getName(),
            nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{DYN_BATCH_MIN, 3, infer_H, infer_W});
        profile->setDimensions(
            network->getInput(0)->getName(),
            nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{DYN_BATCH_OPT, 3, infer_H, infer_W});
        profile->setDimensions(
            network->getInput(0)->getName(),
            nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{DYN_BATCH_MAX, 3, infer_H, infer_W});
        
        int calibBatch = DYN_BATCH_OPT;     //calib use OPT batch
        printf("[MODE] Dynamic batch  min=%d  opt=%d  max=%d  H=%d  W=%d\n",
           DYN_BATCH_MIN, DYN_BATCH_OPT, DYN_BATCH_MAX, infer_H, infer_W);
    #else
        //static batch
        profile->setDimensions(
            network->getInput(0)->getName(),
            nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{STATIC_BATCH, 3, infer_H, infer_W});
        profile->setDimensions(
            network->getInput(0)->getName(),
            nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{STATIC_BATCH, 3, infer_H, infer_W});
        profile->setDimensions(
            network->getInput(0)->getName(),
            nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{STATIC_BATCH, 3, infer_H, infer_W});
        int calibBatch = STATIC_BATCH;
        printf("[MODE] Static batch  batch=%d  H=%d  W=%d\n", STATIC_BATCH, infer_H, infer_W);
    #endif

    /*
    *=======Builder Config==========
    */
   nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
   config->addOptimizationProfile(profile);
   config->setBuilderOptimizationLevel(5);

   // precision: FP16 + INT8
   config->clearFlag(nvinfer1::BuilderFlag::kTF32);
   config->setFlag(nvinfer1::BuilderFlag::kFP16);
   config->setFlag(nvinfer1::BuilderFlag::kINT8);

   /*
   *=======INT8 Calib======
   */
   Int8EntropyCalibrator calibrator(calibBatch, sizeList, infer_H, infer_W, calibrationImagesDir, cacheFile, network->getInput(0)->getName());
   config->setInt8Calibrator(&calibrator);
   getInfoOfLaunchedCommand(pathToOnnx, pathToEngine, cacheFile, profile, network, config);


   /*
   * ========save Engine ======
   */
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    std::ofstream engine_file(pathToEngine, std::ios::binary);
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write(static_cast<char*>(plan->data()), plan->size());
    engine_file.close();

    printf("Engine saved → %s\n", pathToEngine);

    delete plan;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return 0;
}
