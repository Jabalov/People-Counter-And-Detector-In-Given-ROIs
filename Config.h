#include <iostream>


struct Config
{
	float confidencefThresh;
	float nonMaximumSupThresh;
	int inputWidth;
	int inputHeight;
	std::string classesFile;
	std::string modelConfiguration;
	std::string modelWeights;
	std::string networkName;
};


