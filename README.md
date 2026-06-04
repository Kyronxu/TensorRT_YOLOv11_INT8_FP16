# TensorRT_YOLOv11_INT8_FP16
针对YOLOv11进行fp16和int8量化，显著提升推理速度(C++) (包含完整模型转换流程和代码）



### 写在前面

本项目以 **YOLOv11 目标检测模型** 为例，首先将预训练权重 `yolo11s.pt` 通过指令或脚本转换为 `.onnx` 格式；随后采用 **训练后量化（Post-Training Quantization，PTQ）** 的方式，将 ONNX 模型进一步转换为 **INT8 精度的 TensorRT `.engine` 文件**。最后，在经过优化的 **C++ 版本 TensorRT YOLOv11 推理框架**中加载该 engine 文件进行推理，从而显著提升模型的推理速度。



### 测试环境(Nvidia Jetson Orin Nx 16g) and (Nvida GeForce RTX 4070)

- Ultralytics-8.3.225
- Python 3.8
- CUDA:11.8
- CuDNN:8.9.7
- TensorRT:8.6.1.6
- OpenCV:4.10.0



### (1) 从`.pt`到`.onnx`

为了将在Pytorch训练好的`.pt`模型转化到TensorRT所需要的`.engine`模型，通常是需要进行一步中间转换，需要将Pytorch训练好的模型转换成`.onnx`格式。本项目为了能够更快速地实现整体的转换过程，因此选择`yolo11s.pt`模型作为转换模型，在`Ultralytics-8.3.225`版本的Ultralytics中进行模型转换。为了避免文字过于冗长，关于`Ultralytics`的安装这里不再赘述，请按照`Ultralytics`官方指导进行安装，安装的版本不固定，但是为了确保转换的成功率，请尽量和本文的版本保持一致。另外进行转换的模型可以从[这里](https://github.com/ultralytics/assets/releases)找到并下载到。

将模型从`.pt`到`.onnx`的转换相对简单，如果想让**输入的batch是固定**的话，可以在进入文件目录的命令行中执行以下指令：

```shell
$ yolo export model=yolo11s.pt format=onnx imgsz=640
```

命令行执行完毕后便可以成功将`yolo11s.pt`转换成`yolo11s.onnx`了，可以将其用于后续的模型转换。

如果希望**输入的batch是动态的**，可以在`Ultralytics`的根目录下新建一个名为`onnx_exporter.py`的转换文件，文件中可以粘贴如下代码：

```python
from ultralytics import YOLO
import onnx
from onnx import helper

# 第一步：用ultralytics导出（dynamic=True，全动态）
model = YOLO("yolo11s.pt")
model.export(format='onnx', imgsz=640, dynamic=True, opset=17)

# 第二步：加载并修改，固定H/W为640，只保留batch动态
model_onnx = onnx.load("yolo11s.onnx")

for inp in model_onnx.graph.input:
    shape = inp.type.tensor_type.shape
    for i, dim in enumerate(shape.dim):
        if i == 0:
            dim.dim_param = "batch"  # batch保持动态
        elif i == 2:
            dim.dim_param = ""
            dim.dim_value = 640      # H固定640
        elif i == 3:
            dim.dim_param = ""
            dim.dim_value = 640      # W固定640

onnx.save(model_onnx, "yolo11s_dynamic_batch.onnx")
print("Done!")

# 验证
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("yolo11s_dynamic_batch.onnx")
for batch in [1, 4, 6, 8]:
    x = np.random.rand(batch, 3, 640, 640).astype(np.float32)
    out = session.run(None, {"images": x})
    print(f"batch={batch} → output: {out[0].shape}")
```

上述代码的作用是让转换后的`.onnx`文件的batch不被指定，可以在后续使用的时候再进行指定。

随后，可以通过如下的脚本将`.pt`模型转换成`.onnx`模型，转换后的模型名字为`yolo11s_dynamic_batch.onnx`：

```sh
$python onnx_exporter.py
```



### (2)从`.onnx`到`.engine`

为了能够在包含Nvidia显卡的平台上让算法获得更快的推理速度，还需要将模型从`.onnx`转换到TensorRT所需要的`.engine`格式。

为了能够让模型能够在TensorRT框架下推理得更快，通常会将其转化成`fp16`的精度，甚至为了获得最快的推理的推理速度，会以牺牲一部分精度的代价，尝试将模型以`int8`的精度进行推理。

- [x] 将`.onnx`模型转换成`fp16`精度的`.engine`:

  将模型转换成`fp16`精度相对简单，只需要通过以下的指令便可完成：

  ```sh
  # 固定batch
  trtexec \
    --onnx=yolo11s.onnx \
    --saveEngine=yolo11s_fp16.engine \
    --fp16 \
    --memPoolSize=workspace:4096 \
    --verbose
  
  # 非固定batch
  trtexec \
    --onnx=yolo11s_dynamic_batch.onnx \
    --saveEngine=yolo11s_dynamic_fp16.engine \
    --fp16 \
    --memPoolSize=workspace:4096 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:6x3x640x640 \
    --maxShapes=images:12x3x640x640 \
    --verbose
  ```

- [x] 将`.onnx`模型转换成`int8`精度的`.engine`:

  对训练好的模型进行`.int8`量化，通常有两种方式，一种是`PTQ`（训练后量化），另一种是`QAT`（训练中模拟量化。关于这两种量化的实现原理，这里不过多解释，感兴趣可以自行搜索。由于`PTQ`这种量化方式相对简单且易用，本项目就以该方式进行`.int8`量化。

  > PTQ 的核心思想是：
  >  **使用少量标定数据统计模型各层的数值范围，然后计算量化比例，将 FP32 权重和激活映射为 INT8，从而生成更快、更小的推理模型。**

  使用`PTQ`进行模型的`.int8`量化，需要准备标定数据，由于`YOLO11`是在COCO数据集上进行训练的，这里使用`COCO128`数据集进行标定（虽然可能数据不太够，影响最后的推理精度）。`COCO128`数据集可以从[这里](https://www.kaggle.com/datasets/ultralytics/coco128)下载到。

  **数据标定和模型程序**存放到`int8_calibrator_cpp`文件夹下，该项目参考自[int8_calibrator_cpp](https://github.com/Egorundel/int8_calibrator_cpp)，本项目中对`main.cpp`做了一个简单修改，使其可以支持**固定batch和动态batch**的输入。

  为了能成功完成模型的`int8`量化，需要你提前对`main.cpp`做以下修改：

  ```c++
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
  
  ```

  以下是编译脚本操作（需要提前修改好自己的CMakeLists.txt)：

  ```sh
  $ mkdir build & cd build
  $ cmake ..
  $ make -j8
  $ ./int8_calibrator_cpp
  ```

  需要花费以上会需要花费较长的时间进行模型的`int8`量化，量化后会生成`.engine`文件。

### (3) C++推理`.engine`模型，验证推理速度

**模型推理程序**存放在`yolo11`文件夹下，该推理的C++实现，是对本人另一个项目[YOLOv8_TensorRT_Jetson](https://github.com/zhahoi/YOLOv8_TensorRT_Jetson)的修改。在`yolov11`文件夹下可以对其进行编译，用于推理速度验证。

编译脚本如下(（需要提前修改好自己的CMakeLists.txt)：

```sh
$ mkdir build & cd build
$ cmake ..
$ make -j8
$ ./yolov11 xxxxx.engine
```



### `fp16`和`int8`精度`.engine`推理速度和消耗资源对比(固定batch为1测试)

| 指标         | INT8         | FP16         |
| ------------ | ------------ | ------------ |
| Precision    | INT8         | FP16         |
| Engine Size  | **12 MB**    | **23 MB**    |
| GPU Latency  | **3.80 ms**  | **4.31 ms**  |
| Host Latency | **4.20 ms**  | **4.77 ms**  |
| Throughput   | **≈261 FPS** | **≈218 FPS** |
| Enqueue Time | ≈1.30 ms     | ≈1.38 ms     |

YOLOv11模型推理比较

| 项目     | FP16    | INT8   | 提升      |
| -------- | ------- | ------ | --------- |
| 推理时间 | 13.5 ms | 8.8 ms | **1.53×** |
| 总延迟   | 17 ms   | 12 ms  | **1.4×**  |
| FPS      | 58      | 83     | **+43%**  |



### Reference

- [int8_calibrator_cpp](https://github.com/Egorundel/int8_calibrator_cpp)

