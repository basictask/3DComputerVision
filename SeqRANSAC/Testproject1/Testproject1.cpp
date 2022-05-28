#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	std::string image_path = "C://Users//Daniel Kuknyo//OpenCV-4.5.3//opencv//sources//samples//data//starry_night.jpg";
	Mat img = imread(image_path, IMREAD_COLOR);

	if (img.empty())
	{
		cout << "Could not read the image: " << image_path << endl;
		return 1;
	}

	imshow("Display window", img);

	int k = waitKey(0); // Wait for a keystroke in the window
	if (k == 's')
	{
		imwrite("starry_night.png", img);
	}

	return 0;
}