#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>	
#include <fstream>

using namespace cv;
using namespace std;

#define RANSAC_ITER 500
#define THRESHOLD 0.01
#define b 120 // Baseline

struct normPacker
{
	Mat T2Dx;
	Mat T2Dxp;
	vector<pair<Point2f, Point2f>> normedPoints;
};

struct norm2DArray
{
	Mat T;
	vector<Point2f> xp;
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
	strcat_s(param, argv[4]); // ASIFT exe
	strcat_s(param, "\"");
	strcat_s(param, " ");
	strcat_s(param, "\"");
	strcat_s(param, argv[2]); // Left image
	strcat_s(param, "\"");
	strcat_s(param, " ");
	strcat_s(param, "\"");
	strcat_s(param, argv[3]); // Right image
	strcat_s(param, "\"");
	strcat_s(param, " imgOutVert.png imgOutHori.png matchings.txt keys1.txt keys2.txt"); // Params list
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

struct norm2DArray normalization(vector<Point2f> pts) // Normalize a vector of [x, y] points and return the transformed form and T affine transformation vector.
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

	struct norm2DArray result;
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

	struct norm2DArray pts2DxNorm = normalization(pts2Dx); // Normalize primary correspondences
	struct norm2DArray pts2DxpNorm = normalization(pts2Dxp); // Normalize secondary correspondences

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

vector<int> generateRandInd(int n, int num) // Generate n different random integers, in [0, num] range
{
	vector<int> result = vector<int>();

	while (result.size() < n) 
	{
		float randf = (float)(rand()) / RAND_MAX;
		int randi = (int)(randf * num);
		if (result.size() == 0)
		{
			result.push_back(randi);
		}
		else
		{
			bool flag = false;
			for (int i = 0; i < result.size(); i++)
			{
				if (result.at(i) == randi) 
				{
					flag = true;
				}
			}
			if (!flag)
			{
				result.push_back(randi);
			}
		}
	}

	return result;
}

Mat calcFundamentalMatrix(vector<pair<Point2f, Point2f>> pointPairs) // Calculate the fundamental matrix for some samples -> 8-point method
{
	int num = pointPairs.size();
	Mat result; 
	Mat A(num, 9, CV_32F);

	for (int i = 0; i < num; i++)
	{
		float x1 = pointPairs.at(i).first.x;
		float y1 = pointPairs.at(i).first.y;
		float x2 = pointPairs.at(i).second.x;
		float y2 = pointPairs.at(i).second.y;

		A.at<float>(i, 0) = x1 * x2;
		A.at<float>(i, 1) = x2 * y1;
		A.at<float>(i, 2) = x2;
		A.at<float>(i, 3) = y2 * x1;
		A.at<float>(i, 4) = y2 * y1;
		A.at<float>(i, 5) = y2;
		A.at<float>(i, 6) = x1;
		A.at<float>(i, 7) = y1;
		A.at<float>(i, 8) = 1;
	}

	Mat evals;
	Mat evecs;
	Mat AtA = A.t() * A;

	eigen(AtA, evals, evecs);
	Mat x = evecs.row(evecs.rows - 1);
	
	result.create(3, 3, CV_32F);
	memcpy(result.data, x.data, sizeof(float) * 9);

	return result;
}

float pointLineDistance(Point2f pt, Mat L) // Calculates the distance between a point and a line (required in Matrix form)
{
	float aL = (float)L.at<float>(0);
	float bL = (float)L.at<float>(1);
	float cL = (float)L.at<float>(2);

	float tL = abs(aL * pt.x + bL * pt.y + cL);
	float dL = sqrt(aL * aL + bL * bL);
	float dist = tL / dL;

	return dist;
}

float symmetricEpipolarDistance(pair<Point2f, Point2f> pair, Mat F) // Returns the symmetric epipolar distance w.r.t.a  correspondence and a F matrix
{
	Mat pt1 = matFromPoint(pair.first); // Registered point in homogenous representation
	Mat pt2 = matFromPoint(pair.second); // Destination point in homogenous representation

	Mat lL = F.t() * pt2;
	Mat lR = F * pt1;
	
	float distanceL = pointLineDistance(pair.first, lL); 
	float distanceR = pointLineDistance(pair.second, lR);

	float error = 0.5 * (distanceL + distanceR);

	return error;
}

vector<pair<Point2f, Point2f>> removeOutliers(vector<pair<Point2f, Point2f>> pointPairs, vector<int> inlierMask) // Renives outliers based on a vector and a mask
{
	vector<pair<Point2f, Point2f>> result = vector<pair<Point2f, Point2f>>();
	int num = inlierMask.size();
	int inlierCounter = 0;

	for (int i = 0; i < num; i++)
	{
		if (inlierMask.at(i) == 0) // Inlier
		{
			result.push_back(pointPairs.at(i));
			inlierCounter++;
		}
	}

	cout << (num - inlierCounter) << " outliers have been eliminated." << endl << endl;

	return result;
}

vector<int> filterOutliers(vector<pair<Point2f, Point2f>> pointPairs)
{
	int num = pointPairs.size();
	float ythresh = 0.0005;
	int bestInlierNum = -1;
	vector<int> bestInlierMask;

	for (int i = 0; i < RANSAC_ITER; i++)
	{
		int sampleSize = 8;
		vector<int> indices = generateRandInd(sampleSize, num); // Get 8 random indices
		vector<pair<Point2f, Point2f>> ranSample; // 8 randomly chosen correspondences 

		for (int i = 0; i < sampleSize; i++) ranSample.push_back(pointPairs.at(indices.at(i))); // Pick out the random elements

		Mat F = calcFundamentalMatrix(ranSample); // Get Fundamental matrix w.r.t. the sample

		vector<float> distances;
		for (int i = 0; i < num; i++) // Calculate the epipolar distance for each correspondence
		{
			pair<Point2f, Point2f> currpt = pointPairs.at(i);
			if (abs(currpt.first.y - currpt.second.y) > ythresh) // If the two y coordinates don't match --> They must be an outlier
			{
				distances.push_back(THRESHOLD*2); // Has to be classified as outlier: any number larger than t will do
			}
			else
			{
				float epipolarDistance = symmetricEpipolarDistance(currpt, F); // Calculate the epipolar distance based on the correspondence
				distances.push_back(epipolarDistance);
			}
		}

		vector<int> inliers;
		int inlierCount = 0;
		for (int i = 0; i < num; i++) // Decide if a correspondence is an outlier
		{
			if (distances.at(i) < THRESHOLD)
			{
				inliers.push_back(0); // Inlier --> 0
				inlierCount++;
			}
			else
			{
				inliers.push_back(1); // Outlier --> 1
			}
		}

		if (inlierCount > bestInlierNum) // Update the best mask if needed
		{
			bestInlierMask = inliers;
			bestInlierNum = inlierCount;
		}
	}

	return bestInlierMask;
}

Mat getIntrinsicMatrix() // Get intrinsic camera matrix for the images
{
	// This is a hardcoded solution because in this case the intrinsic matrix is not calculated
	Mat result = Mat::zeros(3, 3, CV_32F);
	result.at<float>(0, 0) = 621.18;
	result.at<float>(1, 1) = 621.18;
	result.at<float>(0, 2) = 404;
	result.at<float>(1, 2) = 309;
	result.at<float>(2, 2) = 1;
	return result;
}

vector<Point3f> triangulatePoints(vector<pair<Point2f, Point2f>> pointPairs) // Determine 3D representations of the points
{
	int num = pointPairs.size();
	//Mat K = getIntrinsicMatrix();
	//float f = (K.at<float>(0, 0) + K.at<float>(1, 1)) / 2; // Focal length is determined
	float f = 6;
	vector<Point3f> result;

	for (int i = 0; i < num; i++)
	{
		Point2f p1 = pointPairs.at(i).first;
		Point2f p2 = pointPairs.at(i).second;

		float u1 = p1.x;
		float u2 = p2.x;
		float d = u1 - u2;

		// Transformation for triangulation 
		float Z = (b * f) / d;
		float X = -1 * ((b * (u1 + u2)) / (2 * d));
		float Y = (b * p1.y) / d;


		if (!isinf(X) && !isinf(Y) && !isinf(Z)) // Check for invalid numbers
		{
			Point3f newpt = Point3f(X, Y, Z);
			result.push_back(newpt);
		}
	}

	return result;
}

void outputxyz(vector<Point3f> points, int i) // Outputs a vector of 3D points into a .xyz file
{
	// Get name of file 
	char filename[10];
	std::string s = std::to_string(i);
	char const *pchar = s.c_str();
	strcpy_s(filename, "o");
	strcat_s(filename, "u");
	strcat_s(filename, "t");
	strcat_s(filename, pchar);
	strcat_s(filename, ".");
	strcat_s(filename, "x");
	strcat_s(filename, "y");
	strcat_s(filename, "z");

	// Output file into a file Outx.xyz x=(0,1,..,num)
	int num = points.size();
	ofstream myfile;
	myfile.open(filename);
	myfile << "#Created by Daniel Kuknyo. \n";

	for (int i = 0; i < num; i++) // Iterate over the points and append each coordinate to the file
	{
		Point3f currpt = points.at(i);
		myfile << currpt.x << " " << currpt.y << " " << currpt.z << " " << endl;
	}

	myfile << endl;
	myfile.close();
}

int main(int argc, char** argv)
{
	// Params: matchings.txt left1.jpg right1.jpg demo_ASIFT.exe malaga.mat
	if (argc != 5)
	{
		cout << " Usage: ASIFT.exe img1 img2" << endl;
		return -1;
	}

	//findCorrespondences(argv); // Run ASIFT on a pair of images --> Uncomment to run 

	cout << "> Reading correspondences from file." << endl << endl;
	vector<pair<Point2f, Point2f>> pointPairs = reorgMatches(argv); // Read correspondences into a matrix form
	
	cout << "> Found a total of " << pointPairs.size() << " correspondences" << endl << endl;
	cout << "> Normalizing points" << endl << endl;
	struct normPacker norm = normCorrespData(pointPairs); // Struct for normalizing points
	vector<pair<Point2f, Point2f>> pointPairsNormed = norm.normedPoints; // Normalized points in pair form

	cout << "> Filtering outlier correspondences with robust estimation." << endl << endl;
	vector<int> inlierMask = filterOutliers(pointPairsNormed);
	vector<pair<Point2f, Point2f>> pointPairsFiltered = removeOutliers(pointPairs, inlierMask);

	cout << "> Calculating 3D points" << endl << endl;
	vector<Point3f> pointsTriangulated = triangulatePoints(pointPairsFiltered);

	cout << "> Writing matrix to file" << endl << endl;
	int i = 2; // image ID
	outputxyz(pointsTriangulated, i);

	cout << "Done" << endl << endl;
	return 0;
}