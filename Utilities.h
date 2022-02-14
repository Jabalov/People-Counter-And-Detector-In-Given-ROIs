#include <vector>
#include <opencv2\opencv.hpp>


class Utilities
{

public:
	static std::vector<cv::Mat> readImagesFromPath(std::string path)
	{
		std::vector<cv::String> fn;
		cv::glob(path, fn, false);

		std::vector<cv::Mat> images;
		size_t count = fn.size();
		for(size_t i = 0; i < count; i++)
			images.push_back(cv::imread(fn[i]));
		
		return images;
	}
};