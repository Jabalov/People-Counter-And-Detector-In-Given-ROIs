#pragma once
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Config.h"

class Yolo
{
private:
	cv::dnn::Net neuralNetwork;
	float confidencefThresh;
	float nonMaximumSupThresh;
	int inputWidth;
	int inputHeight;
	char networkName[25];
	std::vector<std::string> classes;
	int counters[3] = {0, 0, 0};

	std::vector<cv::Rect> rois = { {10, 8, 747, 558}, {287, 156, 424, 275}, {27, 129, 203, 160} };

	void postProcessing(cv::Mat& frame, const std::vector<cv::Mat>& outputs);
	void drawPrediction(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

public:
	Yolo(Config);
	void getDetections(cv::Mat& frame);
};