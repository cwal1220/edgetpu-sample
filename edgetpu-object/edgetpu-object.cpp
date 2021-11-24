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
using namespace std;

#define INPUT_WIDTH 300
#define INPUT_HEIGHT 300
#define INPUT_CHANNEL 3

vector<string> ReadLabels(const string& filename) 
{
	ifstream file(filename);
	if(!file)
	{
		return {}; // Open failed.
	}  
	vector<string> lines;
	for (string line; getline(file, line);)
	{
		lines.emplace_back(line);
	}
	return lines;
}

string GetLabel(const vector<string>& labels, int label) 
{
	if (label >= 0 && label < labels.size())
	{
		return labels[label];
	}
	return to_string(label);
}


int get_color(int c, int classNum)
{
	float colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
	int offset = classNum * 123457 % 80;
	float ratio = ((float)classNum / 80) * 5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float ret = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
	return ret * 255;
}


int main(int argc, char* argv[]) 
{
	if (argc != 3) 
	{
		cerr << argv[0] << " <model_file> <label_file>" << endl;
		return 1;
	}
	const string model_file = argv[1];
	const string label_file = argv[2];

	// Find TPU device.
	size_t num_devices;
	unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

	if (num_devices == 0) 
	{
		cerr << "No connected TPU found" << endl;
		return 1;
	}
	const auto& device = devices.get()[0];

	// Load labels.
	auto labels = ReadLabels(label_file);
	if (labels.empty()) 
	{
		cerr << "Cannot read labels from " << label_file << endl;
		return 1;
	}

	// Load model.
	auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
	if (!model) 
	{
		cerr << "Cannot read model from " << model_file << endl;
		return 1;
	}

	// Create interpreter.
	tflite::ops::builtin::BuiltinOpResolver resolver;
	unique_ptr<tflite::Interpreter> interpreter;
	if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) 
	{
		cerr << "Cannot create interpreter" << endl;
		return 1;
	}

	auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
	interpreter->ModifyGraphWithDelegate(delegate);

	// Allocate tensors.
	if (interpreter->AllocateTensors() != kTfLiteOk) 
	{
		cerr << "Cannot allocate interpreter tensors" << endl;
		return 1;
	}

	// Set interpreter input.
	const auto* input_tensor = interpreter->input_tensor(0);
	if (input_tensor->type != kTfLiteUInt8 ||
		input_tensor->dims->data[0] != 1 ||
		input_tensor->dims->data[1] != INPUT_HEIGHT ||
		input_tensor->dims->data[2] != INPUT_WIDTH||
		input_tensor->dims->data[3] != INPUT_CHANNEL)
	{
		cerr << "Input tensor shape does not match input image" << endl;
		return 1;
	}
	
	VideoCapture cap(0);
	cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G') );
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	cap.set(CAP_PROP_FPS, 30);

	int cameraWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	int cameraHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
	int cameraFormat = cap.get(CAP_PROP_FOURCC);
	int cameraFps = cap.get(CAP_PROP_FPS);
	cout << cameraWidth << "x" << cameraHeight << ", fmt:" << cameraFormat << ", fps:" << cameraFps << endl;

	if (!cap.isOpened())
	{
		cout << "Camera Open Fail /dev/video0" << endl;
		return 0;
	}

	Mat cameraFrame, workingFrame;

	namedWindow("Result", WND_PROP_FULLSCREEN);
	setWindowProperty("Result", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
	time_t start, end;
	while(1) 
	{
		start = clock();
		cap >> cameraFrame;
		// cvtColor(cameraFrame, workingFrame, COLOR_BGR2RGB);
		resize(cameraFrame, workingFrame, Size(300, 300));
		copy(workingFrame.data, workingFrame.data + 300*300*3, interpreter->typed_input_tensor<uint8_t>(0));

		// Run inference.
		if (interpreter->Invoke() != kTfLiteOk) 
		{
			cerr << "Cannot invoke interpreter" << endl;
			return 1;
		}

		const float* detection_boxes = interpreter->tensor(interpreter->outputs()[0])->data.f;
		const float* detection_class = interpreter->tensor(interpreter->outputs()[1])->data.f;
		const float* detection_score = interpreter->tensor(interpreter->outputs()[2])->data.f;
		const int    detection_number = *interpreter->tensor(interpreter->outputs()[3])->data.f;

		const float confidence_threshold = 0.5;
		for(int i = 0; i < detection_number; i++)
		{
			if(detection_score[i] > confidence_threshold)
			{
				int det_index = (int)detection_class[i] + 1;
				float y1 = detection_boxes[4*i+0] * cameraHeight;
				float x1 = detection_boxes[4*i+1] * cameraWidth;
				float y2 = detection_boxes[4*i+2] * cameraHeight;
				float x2 = detection_boxes[4*i+3] * cameraWidth;
				Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
				int blue = get_color(0, detection_class[i]);
				int green = get_color(1, detection_class[i]);
				int red = get_color(2, detection_class[i]);

				rectangle(cameraFrame, rec, Scalar(red, green, blue), 2, 8, 0);
				putText(cameraFrame, format("%s", labels[det_index].c_str()), Point(x1, y1-5), FONT_HERSHEY_SIMPLEX, 1, Scalar(red, green, blue), 1, 8, 0);
			}
		}

		end = clock();
		float realFps = 1000.0  / ( (double)(end - start) / 1000);
		putText(cameraFrame, format("%.2f FPS", realFps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 1, 8, 0);
		imshow("Result", cameraFrame);

		// if 'q' button pressed
		if (waitKey(1) == 27)
		{
			break;
		}

	}
	return 0;
}
