#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <time.h>

#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<std::string> ReadLabels(const std::string& filename) {
	std::ifstream file(filename);
	if (!file) return {};  // Open failed.

	std::vector<std::string> lines;
	for (std::string line; std::getline(file, line);) lines.emplace_back(line);
	return lines;
}

std::string GetLabel(const std::vector<std::string>& labels, int label) {
	if (label >= 0 && label < labels.size()) return labels[label];
	return std::to_string(label);
}

std::vector<float> Dequantize(const TfLiteTensor& tensor) {
	const auto* data = reinterpret_cast<const uint8_t*>(tensor.data.data);
	std::vector<float> result(tensor.bytes);
	for (int i = 0; i < tensor.bytes; ++i)
	result[i] = tensor.params.scale * (data[i] - tensor.params.zero_point);
	return result;
}

std::vector<std::pair<int, float>> Sort(const std::vector<float>& scores,
										float threshold) {
	std::vector<const float*> ptrs(scores.size());
	std::iota(ptrs.begin(), ptrs.end(), scores.data());
	auto end = std::partition(ptrs.begin(), ptrs.end(),
							[=](const float* v) { return *v >= threshold; });
	std::sort(ptrs.begin(), end,
			[](const float* a, const float* b) { return *a > *b; });

	std::vector<std::pair<int, float>> result;
	result.reserve(end - ptrs.begin());
	for (auto it = ptrs.begin(); it != end; ++it)
	result.emplace_back(*it - scores.data(), **it);
	return result;
}

int main(int argc, char* argv[]) 
{
	if (argc != 3) 
	{
		std::cerr << argv[0] << " <model_file> <label_file>" << std::endl;
		return 1;
	}

	const std::string model_file = argv[1];
	const std::string label_file = argv[2];

	// Find TPU device.
	size_t num_devices;
	std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
		edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

	if (num_devices == 0) {
	std::cerr << "No connected TPU found" << std::endl;
	return 1;
	}
	const auto& device = devices.get()[0];

	// Load labels.
	auto labels = ReadLabels(label_file);
	if (labels.empty()) {
	std::cerr << "Cannot read labels from " << label_file << std::endl;
	return 1;
	}

	// Load model.
	auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
	if (!model) {
	std::cerr << "Cannot read model from " << model_file << std::endl;
	return 1;
	}

	// Create interpreter.
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
	std::cerr << "Cannot create interpreter" << std::endl;
	return 1;
	}

	auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
	interpreter->ModifyGraphWithDelegate(delegate);

	// Allocate tensors.
	if (interpreter->AllocateTensors() != kTfLiteOk) {
		std::cerr << "Cannot allocate interpreter tensors" << std::endl;
		return 1;
	}

	int image_bpp = 3; 
	int image_width = 224; 
	int image_height = 224;

	// Set interpreter input.
	const auto* input_tensor = interpreter->input_tensor(0);
	if (input_tensor->type != kTfLiteUInt8 ||           //
		input_tensor->dims->data[0] != 1 ||             //
		input_tensor->dims->data[1] != image_height ||  //
		input_tensor->dims->data[2] != image_width ||   //
		input_tensor->dims->data[3] != image_bpp) 
	{
		std::cerr << "Input tensor shape does not match input image" << std::endl;
		return 1;
	}
	VideoCapture cap(0);
	cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G') );
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	cap.set(CAP_PROP_FPS, 30);

	int width = cap.get(CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CAP_PROP_FRAME_HEIGHT);
	int format = cap.get(CAP_PROP_FOURCC);
	int fps = cap.get(CAP_PROP_FPS);
	std::cout << width << "x" << height << ", fmt:" << format << ", fps:" << fps << std::endl;
	if (!cap.isOpened())
	{
		std::cout << "Camera Open Fail /dev/video0" << std::endl;
		return 0;
	}
	Mat cameraFrame, workingFrame;
	namedWindow("Result", 1);

	time_t start, end;
	while(1) 
	{
		start = clock();
		cap >> cameraFrame;
		cvtColor(cameraFrame, workingFrame, COLOR_BGR2RGB);
		resize(workingFrame, workingFrame, Size(224, 224));

		std::copy(workingFrame.data, workingFrame.data+ 224*224*3, interpreter->typed_input_tensor<uint8_t>(0));

		// Run inference.
		if (interpreter->Invoke() != kTfLiteOk) 
		{
			std::cerr << "Cannot invoke interpreter" << std::endl;
			return 1;
		}

		// Get interpreter output.
		int count = 20;
		auto results = Sort(Dequantize(*interpreter->output_tensor(0)), 0.0001);
		for (auto& result : results)
		{
			putText(cameraFrame, GetLabel(labels, result.first) + " : " + std::to_string(result.second), Point(10, count), 2, 0.8, Scalar(128,255,128) );
			count += 20;
			if(count > 120)
			{
				break;
			}
		}

		imshow("Result", cameraFrame);
		// if 'q' button pressed
		if (waitKey(1) == 27)
		{
			break;
		}
		end = clock();
		std::cout << (double)(end - start) << std::endl;
	}
	return 0;
}
