# Yolov5 Yolov4 Yolov3 TensorRT Implementation

![GitHub stars](https://img.shields.io/github/stars/enazoe/yolo-tensorrt) ![GitHub forks](https://img.shields.io/github/forks/enazoe/yolo-tensorrt)  ![GitHub watchers](https://img.shields.io/github/watchers/enazoe/yolo-tensorrt)  [![Gitter](https://badges.gitter.im/yolo-tensorrt/community.svg)](https://gitter.im/yolo-tensorrt/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

__news:__ 2021.10.31:yolov5-v6.0 support

![](./configs/result.jpg)
## INTRODUCTION

The project is the encapsulation  of nvidia official yolo-tensorrt [implementation](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps). And you must have the trained yolo model(__.weights__) and __.cfg__ file from the darknet (yolov3 & yolov4). For the [yolov5](https://github.com/ultralytics/yolov5) ,you should prepare the model file (yolov5s.yaml) and the trained weight file (yolov5s.pt) from pytorch.

![](./configs/yolo-trt.png)

- [x] yolov5n ,yolov5s , yolov5m , yolov5l , yolov5x ,yolov5-p6 [tutorial](yolov5_tutorial.md)
- [x] yolov4 
- [x] yolov3 

## Features

- [x] inequal net width and height
- [x] batch inference
- [x] support FP32,FP16,INT8
- [ ] dynamic input size


## PLATFORM & BENCHMARK

- [x] windows 10
- [x] ubuntu 18.04
- [x] L4T (Jetson platform)

## JetPack 6.0

This branch targets JetPack 6.0 on Jetson Orin only.

- Jetson Linux 36.3 / Ubuntu 22.04
- CUDA 12.2
- TensorRT 8.6
- cuDNN 8.9

The TensorRT runtime has been migrated to explicit-batch engines for JP6. Old JetPack 5.x `.engine`
files are intentionally incompatible with this branch and will be regenerated with a JP6-specific
cache name.

<details><summary><b>BENCHMARK</b></summary>

#### x86 (inference time)


|  model  |  size   |  gpu   | fp32 | fp16 | INT8 |
| :-----: | :-----: | :----: | :--: | :--: | :--: |
| yolov5s | 640x640 | 1080ti | 8ms  |  /   | 7ms  |
| yolov5m | 640x640 | 1080ti | 13ms |  /   | 11ms |
| yolov5l | 640x640 | 1080ti | 20ms |  /   | 15ms |
| yolov5x | 640x640 | 1080ti | 30ms |  /   | 23ms |
#### Jetson NX with Jetpack4.4.1 (inference / detect time)

|      model      |      size      |  gpu   | fp32 | fp16 | INT8 |
| :-------------: | :----: | :--: | :--: | :--: | :--: |
| yolov3 | 416x416 | nx | 105ms/120ms |  30ms/48ms  | 20ms/35ms |
| yolov3-tiny | 416x416 | nx | 14ms/23ms  |  8ms/15ms   | 12ms/19ms  |
| yolov4-tiny | 416x416 | nx | 13ms/23ms  |  7ms/16ms   | 7ms/15ms  |
| yolov4 | 416x416 | nx | 111ms/125ms  |  55ms/65ms  | 47ms/57ms  |
| yolov5s | 416x416 | nx | 47ms/88ms |  33ms/74ms   | 28ms/64ms |
|   yolov5m   | 416x416 | nx | 110ms/145ms |  63ms/101ms   | 49ms/91ms |
| yolov5l | 416x416 | nx | 205ms/242ms |  95ms/123ms   | 76ms/118ms |
| yolov5x | 416x416 | nx | 351ms/405ms |  151ms/183ms   | 114ms/149ms |


### ubuntu 
|      model      |      size      |  gpu   | fp32 | fp16 | INT8 |
| :-------------: | :----: | :--: | :--: | :--: | :--: |
| yolov4 | 416x416 | titanv | 11ms/17ms  |  8ms/15ms  | 7ms/14ms  |
| yolov5s | 416x416 | titanv | 7ms/22ms |  5ms/20ms   | 5ms/18ms |
|   yolov5m   | 416x416 | titanv | 9ms/23ms |  8ms/22ms   | 7ms/21ms |
| yolov5l | 416x416 | titanv | 17ms/28ms |  11ms/23ms   | 11ms/24ms |
| yolov5x | 416x416 | titanv | 25ms/40ms |  15ms/27ms   | 15ms/27ms |
</details>

## WRAPPER

Prepare the pretrained __.weights__ and __.cfg__ model. 

```c++
Detector detector;
Config config;

std::vector<BatchResult> res;
detector.detect(vec_image, res)
```

## Build and use yolo-trt as DLL or SO libraries


### windows10

- dependency : TensorRT 7.1.3.4  , cuda 11.0 , cudnn 8.0  , opencv4 , vs2015
- build:
  
    open MSVC _sln/sln.sln_ file 
    - dll project : the trt yolo detector dll
    - demo project : test of the dll

### ubuntu & L4T (jetson)

The project generate the __libdetector.so__ lib, and the sample code.
**_If you want to use the libdetector.so lib in your own project,this [cmake file](https://github.com/enazoe/yolo-tensorrt/blob/master/scripts/CMakeLists.txt) perhaps could help you ._**


```bash
git clone https://github.com/enazoe/yolo-tensorrt.git
cd yolo-tensorrt/
mkdir build
cd build/
cmake ..
make
./yolo-trt
```

For JetPack 6.0 Orin builds, make sure TensorRT 8.6 is available in the system library path before
running `cmake ..`.
## API

```c++
struct Config
{
	std::string file_model_cfg = "configs/yolov4.cfg";

	std::string file_model_weights = "configs/yolov4.weights";

	std::string engine_file_path = "";

	float detect_thresh = 0.9;

	ModelType net_type = YOLOV4;

	Precision inference_precison = FP32;

	int max_batch_size = 1;

	std::size_t workspace_size_bytes = 1ULL << 30;
	
	int gpu_id = 0;

	std::string calibration_image_list_file_txt = "configs/calibration_images.txt";

};

class API Detector
{
public:
	explicit Detector();
	~Detector();

	void init(const Config &config);

	void detect(const std::vector<cv::Mat> &mat_image,std::vector<BatchResult> &vec_batch_result);

private:
	Detector(const Detector &);
	const Detector &operator =(const Detector &);
	class Impl;
	Impl *_impl;
};
```

Notes:

- `max_batch_size` is now the runtime batch contract for engine generation and inference.
- `workspace_size_bytes` controls the TensorRT workspace pool size on JP6.
- Leave `engine_file_path` empty to use the automatic JP6 cache naming scheme.

## REFERENCE

- https://github.com/wang-xinyu/tensorrtx/tree/master/yolov4
- https://github.com/mj8ac/trt-yolo-app_win64
- https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps

## Contact

微信关注公众号EigenVison，回复yolo获取交流群号



![](./configs/qrcode.jpeg)
