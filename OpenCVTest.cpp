#include <iostream>
#include <vector>
#include "Yolo.h"
#include "Utilities.h"
#include <opencv2\opencv.hpp>
#include <opencv2\video\tracking.hpp>
using namespace std;

	
Config net_config = { 0.5, 0.4, 320, 320, "coco.names", "yolo-fastest-xl.cfg", "yolo-fastest-xl.weights", "yolo-fastest" };

int main(int argc, char** argv)
{
	std:string path = "C:\\Users\\mohab\\Downloads\\Crowd_PETS09\\S1\\L1\\Time_13-57\\\View_001\\*.jpg";

	std::vector<cv::Mat> frames = Utilities().readImagesFromPath(path);
	Yolo NN(net_config);

	int frame_width = frames[0].cols;
	int frame_height = frames[0].rows;
	cv::VideoWriter video("output1.avi", 
						  cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
						  10, 
						  cv::Size(frame_width, frame_height));

	int i = 0;
	while (i < frames.size())
	{
		cv::Mat img = frames[i];
		NN.getDetections(img);

		static const string title = "Object Detection in OpenCV using Yolo-Fastest";
		cv::namedWindow(title, cv::WINDOW_NORMAL);
		cv::imshow(title, img);
		video.write(img);

		if(cv::waitKey(25) >= 0) break;
		i += 2;
	}

	video.release();
	cv::destroyAllWindows();
	return 0;
}