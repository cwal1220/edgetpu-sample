# edgetpu-sample
Google Coral Edge TPU Sample Code(TensorFlow Lite C++, OpenCV)

## Installation
cmake .  
make  

## Usage
./edgetpuclass mobilenet_v1_1.0_224_quant_edgetpu.tflite imagenet_labels.txt  
./edgetpuobject mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite coco_label.txt  

## Tested
- Telechips TCC805x SoC  
- Broadcom BCM2711 Soc(Raspberry Pi 4)  
