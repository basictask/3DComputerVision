#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include "MatrixReaderWriter.h"

using namespace cv;
using namespace std;

#define SQRT2 1.41
#define THRESHOLD 0.001 // minimum symmetric transfer error in pixels  
#define RANSAC_ITER  600    

struct normSingleArray 
{
	Mat T;
	vector<Point2f> xp;
};

struct normPacker
{
	Mat T2Dx;
	Mat T2Dxp;
	vector<pair<Point2f, Point2f>> normedPoints;
};

Mat matFromPoint(Point2f p) // Convert a point into a homogenously represented matrix [x, y, w]^T
{
	Mat pt(3, 1, CV_32F);
	pt.at<float>(0, 0) = p.x;
	pt.at<float>(1, 0) = p.y;
	pt.at<float>(2, 0) = 1.0;
	return pt;
}

void findCorrespondences(char** argv) // Generate a dynamic param and run ASIFT
{
	char param[500];

	strcpy_s(param, "\"");
	strcat_s(param, "\"");
	strcat_s(param, argv[4]);
	strcat_s(param, "\"");
	strcat_s(param, " ");
	strcat_s(param, "\"");
	strcat_s(param, argv[2]);
	strcat_s(param, "\"");
	strcat_s(param, " ");
	strcat_s(param, "\"");
	strcat_s(param, argv[3]);
	strcat_s(param, "\"");
	strcat_s(param, " imgOutVert.png imgOutHori.png matchings.txt keys1.txt keys2.txt");
	strcat_s(param, "\"");

	system(param); // Run ASIFT Correspondence fidner
}

vector<float> splitString(string arg, char splitter) // Split a string on given character and return as an array
{
	stringstream test(arg);
	string segment;
	vector<float> seglist;

	while (std::getline(test, segment, splitter))
	{
		if (segment != "")
		{
			seglist.push_back(std::stof(segment));
		}
	}
	return seglist;
}

vector<pair<Point2f, Point2f>> reorgMatches(char** argv) // Organize correspondences into pair from from txt file output by ASIFT
{
	vector<pair<Point2f, Point2f>> pointPairs;
	fstream newfile;
	newfile.open(argv[1], ios::in);

	if (newfile.is_open())
	{
		string tp;
		while (getline(newfile, tp))
		{
			vector<float> data = splitString(tp, '  ');
			if (data.size() == 4)
			{
				pair<Point2f, Point2f> currPts;

				currPts.first = Point2f((float)data[2], (float)data[3]);
				currPts.second = Point2f((float)data[0], (float)data[1]);
				
				pointPairs.push_back(currPts);
			}
		}
		newfile.close();
	}
	else
	{
		cout << "Cannot open file matchings.txt" << endl; 
		return pointPairs;
	}
	return pointPairs;
}

vector<int> generateRandInt(int num) // Generates 4 random and different indices for robust estimation and returns a 4-element vector.
{
	float rand1 = (float)(rand()) / RAND_MAX;
	float rand2 = (float)(rand()) / RAND_MAX;
	float rand3 = (float)(rand()) / RAND_MAX;
	float rand4 = (float)(rand()) / RAND_MAX;
	
	int index1 = (int)(rand1*num);
	
	int index2 = (int)(rand2*num);
	while (index2 == index1)
	{
		rand2 = (float)(rand()) / RAND_MAX;
		index2 = (int)(rand2*num);
	}
	
	int index3 = (int)(rand3*num);
	while ((index3 == index1) || (index3 == index2))
	{
		rand3 = (float)(rand()) / RAND_MAX;
		index3 = (int)(rand3*num);
	}

	int index4 = (int)(rand4*num);
	while ((index4 == index1) || (index3 == index2) || (index4 == index3))
	{
		rand4 = (float)(rand()) / RAND_MAX;
		index4 = (int)(rand4*num);
	}

	vector<int> result;

	result.push_back(index1);
	result.push_back(index2);
	result.push_back(index3);
	result.push_back(index4);

	return result;
}

float transferError(pair<Point2f, Point2f> corr, Mat H, Mat T2Dx, Mat T2Dxp) // Calculate symmetric error for a single correspondence based on H. 
{
	// Symmetric transfer error between 2 points x and x': d^2T = d(x, inv(H)*x')^2 + d(x', H*x)^2
	Mat x = matFromPoint(corr.first); // Homogenous representation
	Mat xp = matFromPoint(corr.second);
	Mat hx = H * x;
	Mat hxp = H.inv() * xp;

	// Turn back into physical representation
	hx = hx * (1.0 / hx.at<float>(2, 0));
	hxp = hxp * (1.0 / hxp.at<float>(2, 0));

	Point2f p1 = Point2f(x.at<float>(0, 0), x.at<float>(1, 0));     // x 
	Point2f p2 = Point2f(hxp.at<float>(0, 0), hxp.at<float>(1, 0)); // inv(H) * xp
	Point2f p3 = Point2f(xp.at<float>(0, 0), xp.at<float>(1, 0));   // xp
	Point2f p4 = Point2f(hx.at<float>(0, 0), hx.at<float>(1, 0));   // H * x

	float xDist = float(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
	float xpDist = float(pow(p4.x - p3.x, 2) + pow(p4.y - p3.y, 2));

	float res = xDist + xpDist;
	
	return res;
}

vector<pair<Point2f, Point2f>> removeOutliers(vector<pair<Point2f, Point2f>> pointPairs, vector<int> inlierMask) // Removes elements from an array at the indices where mask = 0
{
	vector<pair<Point2f, Point2f>> result;

	for (int i = 0; i < pointPairs.size(); i++)
	{
		if (inlierMask.at(i) == 1)
		{
			result.push_back(pointPairs.at(i));
		}
	}
	return result;
}

struct normSingleArray normalization(vector<Point2f> pts) // Normalize a vector of [x, y] points and return the transformed form and T affine transformation vector.
{
	Point2f mean(0, 0);
	int num = pts.size();
	for (int i = num; i < num; i++)
	{
		mean.x += pts.at(i).x;
		mean.y += pts.at(i).y;
	}

	mean.x /= (float)num;
	mean.y /= (float)num;

	Point2f spread(0, 0);
	for (int i = 0; i < num; i++)
	{
		spread.x += pow(pts.at(i).x - mean.x, 2);
		spread.y += pow(pts.at(i).y - mean.y, 2);
	}

	spread.x = sqrt(2.0) / sqrt(spread.x / num);
	spread.y = sqrt(2.0) / sqrt(spread.y / num);

	float tx = -spread.x * mean.x;
	float ty = -spread.y * mean.y;

	Mat T = Mat::zeros(3, 3, CV_32F);
	T.at<float>(0, 0) = spread.x;
	T.at<float>(1, 1) = spread.y;
	T.at<float>(0, 2) = tx;
	T.at<float>(1, 2) = ty;
	T.at<float>(2, 2) = 1.0;

	// Transform x' = Tx
	vector<Point2f> xpvec;
	for (int i = 0; i < num; i++)
	{
		Mat mx = matFromPoint(pts.at(i));
		Mat mxp = T * mx;
		Point2f xp;

		xp.x = mxp.at<float>(0, 0) / mxp.at<float>(2, 0);
		xp.y = mxp.at<float>(1, 0) / mxp.at<float>(2, 0);

		xpvec.push_back(xp);
	}

	struct normSingleArray result;
	result.xp = xpvec;
	result.T = T;
	return result;
}

struct normPacker normCorrespData(vector<pair<Point2f, Point2f>> pointPairs) // Function to normalize a vector of correspondences
{
	vector<Point2f> pts2Dx; // Template points 
	vector<Point2f> pts2Dxp; // Registered points

	for (int i = 0; i < pointPairs.size(); i++) // Arrange pairs into separate vectors of [x, y] points
	{
		pts2Dx.push_back(pointPairs[i].first);
		pts2Dxp.push_back(pointPairs[i].second);
	}

	struct normSingleArray pts2DxNorm = normalization(pts2Dx); // Normalize primary correspondences
	struct normSingleArray pts2DxpNorm = normalization(pts2Dxp); // Normalize secondary correspondences

	vector<pair<Point2f, Point2f>> normedPoints;

	for (int i = 0; i < pts2Dx.size(); i++)  // Arrange in pair<x, x'> form
	{
		pair<Point2f, Point2f> newpt;

		newpt.first = pts2DxNorm.xp.at(i);
		newpt.second = pts2DxpNorm.xp.at(i);

		normedPoints.push_back(newpt);
	}

	struct normPacker result;
	result.normedPoints = normedPoints;
	result.T2Dx = pts2DxNorm.T;
	result.T2Dxp = pts2DxpNorm.T;
	return result;
}

Mat calcHomography(vector<pair<Point2f, Point2f>> pointPairs)
{
	const int ptsNum = pointPairs.size(); // Number of correspondences.
	Mat A(2 * ptsNum, 9, CV_32F); // A matrix is formed. 2 rows for each correspondence. 32-bit float.

	for (int i = 0; i < ptsNum; i++)
	{
		// Retrieve coordinates for the vectors. Index 1: 1st image, Index 2: coordinates for the 2nd image.
		float u1 = pointPairs[i].first.x; // ui
		float v1 = pointPairs[i].first.y; // vi

		float u2 = pointPairs[i].second.x; // ui'
		float v2 = pointPairs[i].second.y; // vi'

		// Set elements in the odd and even rows in Ai matrix [[ui, vi, 1, 0, 0, 0, -ui*ui' -vi*ui', -ui'], [0, 0, 0, ui, vi, 1, -ui*vi' -vi*vi', -vi']]
		A.at<float>(2 * i, 0) = u1;
		A.at<float>(2 * i, 1) = v1;
		A.at<float>(2 * i, 2) = 1.0f;
		A.at<float>(2 * i, 3) = 0.0f;
		A.at<float>(2 * i, 4) = 0.0f;
		A.at<float>(2 * i, 5) = 0.0f;
		A.at<float>(2 * i, 6) = -u2 * u1;
		A.at<float>(2 * i, 7) = -u2 * v1;
		A.at<float>(2 * i, 8) = -u2;

		A.at<float>(2 * i + 1, 0) = 0.0f;
		A.at<float>(2 * i + 1, 1) = 0.0f;
		A.at<float>(2 * i + 1, 2) = 0.0f;
		A.at<float>(2 * i + 1, 3) = u1;
		A.at<float>(2 * i + 1, 4) = v1;
		A.at<float>(2 * i + 1, 5) = 1.0f;
		A.at<float>(2 * i + 1, 6) = -v2 * u1;
		A.at<float>(2 * i + 1, 7) = -v2 * v1;
		A.at<float>(2 * i + 1, 8) = -v2;
	}

	// Compute the least eigenvector.
	Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);
	cv::eigen(A.t() * A, eVals, eVecs); // Pick the last element --> eigenvector corresponding to the least eigenvalue. 

	Mat H(3, 3, CV_32F);
	for (int i = 0; i < 9; i++)
	{
		H.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i); // Retrieve element for the homography matrix and stack + reorder elements
	}

	H = H * (1.0 / H.at<float>(2, 2));

	return H;
}

vector<pair<Point2f, Point2f>> filterRansac(vector<pair<Point2f, Point2f>> pointPairs, Mat T2Dx, Mat T2Dxp)
{
	int num = pointPairs.size();
	int bestInlierNum = -1;
	vector<int> bestInlierMask;

	for (int iter = 0; iter < RANSAC_ITER; iter++)
	{
		vector<int> indices = generateRandInt(num); // Get 4 random indices
		vector<pair<Point2f, Point2f>> ranSample; // The 4 randomly chosen correspondences 
		
		for (int i = 0; i < 4; i++)	ranSample.push_back(pointPairs.at(indices.at(i)));
		
		Mat H = calcHomography(ranSample);

		// Calculate the error for each putative correspondence
		vector<float> distances;
		for (int i = 0; i < num; i++)
		{
			float correspError = transferError(pointPairs.at(i), H, T2Dx, T2Dxp); // Symmetric transfer error w.r.t. the homography matrix given by the random sampling
			distances.push_back(correspError);
		}

		// Decide if a given correspondence is an outlier
		int inlierCount = 0;
		vector<int> inliers;
		for (int i = 0; i < num; i++) // Create mask for inliers
		{ 
			if (distances.at(i) < THRESHOLD)
			{
				inliers.push_back(1); // inlier: 1
				inlierCount++;
			}
			else 
			{
				inliers.push_back(0); // outlier: 0
			}
		}

		if (inlierCount > bestInlierNum) // Update max. variables
		{
			bestInlierMask = inliers;
			bestInlierNum = inlierCount;
		}
	}

	vector<pair<Point2f, Point2f>> result = removeOutliers(pointPairs, bestInlierMask); // Remove all outliers based on [0,1] mask

	cout << pointPairs.size() - result.size() << " outliers have been eliminated. " << endl;

	return result;
}

void transformImage(Mat origImg, Mat& newImage, Mat tr, bool isPerspective)
{
	Mat invTr = tr.inv();

	const int WIDTH = origImg.cols;
	const int HEIGHT = origImg.rows;

	const int newWIDTH = newImage.cols;
	const int newHEIGHT = newImage.rows;

	for (int x = 0; x < newWIDTH; x++)
	{
		for (int y = 0; y < newHEIGHT; y++)
		{
			Mat pt = matFromPoint(Point2f(x, y));

			Mat ptTransformed = invTr * pt;
			if (isPerspective)
			{
				ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;
			}

			int newX = round(ptTransformed.at<float>(0, 0));
			int newY = round(ptTransformed.at<float>(1, 0));

			if ((newX >= 0) && (newX < WIDTH) && (newY >= 0) && (newY < HEIGHT))
			{
				newImage.at<Vec3b>(y, x) = origImg.at<Vec3b>(newY, newX);
			}
		}
	}
}

string convertToString(char* a, int size)
{
	// Converts an array of chars into a single string
	int i;
	string s = "";
	for (i = 0; i < size; i++) {
		s = s + a[i];
	}
	return s;
}

string getImageName(int i, char* firstImage)
{
	// Get the name of the next image. Rule: xxyyzz1.png, xxyyzz2.png, ..., xxyyzzn.png, for i = 0 -> 9
	char *arr_ptr = &firstImage[0];

	int num = strlen(arr_ptr);

	char param[500];

	strcpy_s(param, firstImage);
	
	std::string tmp = std::to_string(i);
	
	char const *num_char = tmp.c_str();
	
	param[num-5] = *num_char;
	
	string ret = convertToString(param, num);

	return ret;
}

void generateImages(char** argv, Mat H, int numImages)
{
	Mat hNormEye = Mat::eye(3, 3, CV_32F);

	for (int i = 0; i < numImages; i++)
	{
		Mat image1 = imread(getImageName(i, argv[2])); // Left image: dev1_0.png ... dev1_9.png
		Mat image2 = imread(getImageName(i, argv[3])); // Right image: dev2_0.png ... dev2_9.png 

		Mat transformedImage = Mat::zeros(1.5 * image1.size().height, 1.8 * image1.size().width, image1.type());

		if (!image2.data || !image1.data) // Check if both images can be read
		{
			throw std::invalid_argument("Could not open or find an image");
		}

		transformImage(image1, transformedImage, hNormEye, true); // Left image
		transformImage(image2, transformedImage, H, true); // Right image

		// Create dynamic image name
		char param[500];
		strcpy_s(param, "panoStitch");
		std::string tmp = std::to_string(i);
		char const *num_char = tmp.c_str();
		strcat_s(param, num_char);
		strcat_s(param, ".png");
		
		// Save image
		imwrite(param, transformedImage);
		cout << "Finished transforming and saving batch " << i << endl;
	}
	cout << "Done stitching all images." << endl;
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        cout << " Usage: ASIFT.exe img1 img2" << endl;
        return -1;
    }

	findCorrespondences(argv); // Run ASIFT on the first 2 images

	cout << "Transforming correspondences." << endl << endl;
    vector<pair<Point2f, Point2f>> pointPairs = reorgMatches(argv);

	cout << "Applying normalization." << endl;
	struct normPacker norm = normCorrespData(pointPairs);
	vector<pair<Point2f, Point2f>> pointPairsNormed = norm.normedPoints; // Normalized points
	Mat T2Dx = norm.T2Dx; // Primary affine transformation matrix 
	Mat T2Dxp = norm.T2Dxp; // Secondary affine transformation matrix

	cout << "Filtering correspondences with robust estimation." << endl;
	pointPairs = filterRansac(pointPairsNormed, T2Dx, T2Dxp);

	if (pointPairs.size() < 4) 
	{
		cout << "Not enough inlier correspondences were found. Exiting.";
		return -1;
	}

	cout << "Calculating normalized Homography matrix." << endl << endl;
    Mat HTx = calcHomography(pointPairs); 
	Mat H = T2Dxp.inv() * HTx * T2Dx;

	cout << "H:" << endl << H << endl << endl;
	cout << "Transforming images." << endl;

	try
	{
		generateImages(argv, H, 10);
	}
	catch (std::invalid_argument& e)
	{
		cerr << e.what() << endl;
		return -1;
	}

	cout << "Done without errors." << endl;
	cv::waitKey(0); 
    return 0;
}