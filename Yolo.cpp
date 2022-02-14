#include "Yolo.h"

Yolo::Yolo(Config conf)
{
	this->confidencefThresh = conf.confidencefThresh;
	this->nonMaximumSupThresh = conf.nonMaximumSupThresh;
	this->inputHeight = conf.inputHeight;
	this->inputWidth = conf.inputWidth;

	strcpy_s(this->networkName, conf.networkName.c_str());

	std::ifstream ifs(conf.classesFile.c_str());
	std::string line;

	while(std::getline(ifs, line)) 
		this->classes.push_back(line);

	this->neuralNetwork = cv::dnn::readNetFromDarknet(conf.modelConfiguration, conf.modelWeights);
	this->neuralNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	this->neuralNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void Yolo::getDetections(cv::Mat& frame)
{
	std::fill(counters, counters + 3, 0);
	cv::Mat blob;
	cv::dnn::blobFromImage(frame, blob, 1 / 255.0, 
						   cv::Size(this->inputWidth, 
						   this->inputHeight), 
						   cv::Scalar(0, 0, 0), 
						   true, false);

	this->neuralNetwork.setInput(blob);
	std::vector<cv::Mat> outputs;

	this->neuralNetwork.forward(outputs, this->neuralNetwork.getUnconnectedOutLayersNames());
	this->postProcessing(frame, outputs);

	for(int i = 0; i < 3; i++)
	{
		cv::rectangle(frame,
			cv::Point(this->rois[i].x, this->rois[i].y),
			cv::Point(this->rois[i].x + this->rois[i].width, this->rois[i].y + this->rois[i].height),
			cv::Scalar(0, 255, 0));


		std::string label = cv::format("Persons Counts in ROI%d: %d", i+1, this->counters[i]);
		cv::putText(frame, 
					label, 
					cv::Point(this->rois[i].x, this->rois[i].y + 15), 
					cv::FONT_HERSHEY_SIMPLEX, 
					0.6, 
					cv::Scalar(0, 0, 255), 
					1.3);
	}
		
}

void Yolo::drawPrediction(int classId, 
						  float conf, 
						  int left, 
						  int top, 
						  int right, 
						  int bottom, 
						  cv::Mat& frame)
{
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

	std::string label = cv::format("%.2f", conf);
	if(!this->classes.empty())
	{
		CV_Assert(classId < (int)this->classes.size());
		label = this->classes[classId] + ":" + label;
	}

	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = std::max(top, labelSize.height);
	putText(frame, 
			label, 
			cv::Point(left, top), 
			cv::FONT_HERSHEY_SIMPLEX, 
			0.75, cv::Scalar(0, 255, 0), 1);
}

void Yolo::postProcessing(cv::Mat& frame, 
						 const std::vector<cv::Mat>& outputs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for(size_t i = 0; i < outputs.size(); ++i)
	{
		float* data = (float*)outputs[i].data;
		for(int j = 0; j < outputs[i].rows; ++j, data += outputs[i].cols)
		{
			cv::Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols);
			cv::Point classIdPoint;
			double confidence;

			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if(confidence > this->confidencefThresh)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, this->confidencefThresh, this->nonMaximumSupThresh, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		if(classIds[idx] == 0) // If the current object class is person and the object is in one of the roi.
		{
			for(int j = this->rois.size() - 1; j >= 0; j--)
			{
				cv::Rect current_roi = this->rois[j];

				if((current_roi.x < box.x + box.width / 2 && current_roi.x + current_roi.width > box.x + box.width / 2)
					&& (current_roi.y < box.y + box.height / 2 && current_roi.y + current_roi.height > box.y + box.height / 2))
				{
					this->counters[j]++;
					this->drawPrediction(classIds[idx],
										 confidences[idx],
										 box.x, 
										 box.y,
										 box.x + box.width, 
										 box.y + box.height, 
										 frame);
					break;
				}
			}
		}
		else continue;
	}
}