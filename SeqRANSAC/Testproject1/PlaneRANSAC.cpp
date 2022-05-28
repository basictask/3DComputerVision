#include "MatrixReaderWriter.h"
#include "PlaneEstimation.h"
#include "PLYWriter.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>       
#include <sstream>

using namespace cv;

#define THERSHOLD 0.2  //RANSAC threshold 
#define RANSAC_ITER  600    //RANSAC iteration
#define FILTER_LOWEST_DISTANCE 0.3 //threshold for pre-filtering

vector<Point3f> removeInliers(vector<Point3f> pointsToRemove) // Remove elements with 0 in every coordinate from a vector
{
	vector<Point3f> result;
	int len = pointsToRemove.size();

	for (int i = 0; i < len; i++)
	{
		Point3f p = pointsToRemove[i];
		if (p.x != 0.0 && p.y != 0.0 && p.z != 0.0)
		{
			result.push_back(p);
		}
	}
	return result;
}

vector<string> splitString(const char* arg, char splitter) // Split a string on given character and return as an array
{
	stringstream test(arg);
	string segment;
	vector<string> seglist;

	while (std::getline(test, segment, splitter))
	{
		seglist.push_back(segment);
	}

	return seglist;
}

string getFileName(vector<string> splitPath) // Get a file's name from a split string vector: last element xxxxx.xyz
{
	int len = splitPath.size();

	char const *file = splitPath[len - 1].c_str();

	vector<string> fileName = splitString(file, '.');

	string name = fileName[0].c_str();

	return name;
}

int main(int argc, char** argv) {

	if (argc != 3) {
		printf("Usage:\n PlanRansac input.xyz output.ply\n");
		exit(EXIT_FAILURE);
	}

	MatrixReaderWriter mrw(argv[1]);

	int num = mrw.rowNum;

	cout << "Rows:" << num << endl;
	cout << "Cols:" << mrw.columnNum << endl;

	vector<Point3f> points;

	for (int idx = 0; idx < num; idx++) {
		double x = mrw.data[3 * idx];
		double y = mrw.data[3 * idx + 1];
		double z = mrw.data[3 * idx + 2];
		float distFromOrigo = sqrt(x * x + y * y + z * z);
		        
		if (distFromOrigo > FILTER_LOWEST_DISTANCE) {
			Point3f newPt;
			newPt.x = x;
			newPt.y = y;
			newPt.z = z;
			points.push_back(newPt);
		}
	}
	num = points.size(); //Number of points:

	vector<Point3f> points_dec = points;	// The array where the inliers are taken out: on first run there are no in/outliers
	vector<Point3i> colorsRANSAC;			// Contains the colors for the planes
	vector<Point3f> pointsToRemove;			// Contains the inlier elements flagged for a specific method to remove
	vector<Point3i> planeColors = {Point3i(174, 4, 33), Point3i(172, 255, 36), Point3i(255, 16, 223)}; //3 colors: Red, Green, Pink

	int planeColorCount = 0;
	int num_planes = 3;
	bool colorFlagPush = true;
	for (int i = 0; i < num_planes; i++)
	{
		float* planeParams = EstimatePlaneRANSAC(points_dec, THERSHOLD, RANSAC_ITER);	    
		printf("Plane params RANSAC:\n A:%f B:%f C:%f D:%f \n", planeParams[0], planeParams[1], planeParams[2], planeParams[3]);

		RANSACDiffs differences = PlanePointRANSACDifferences(points, planeParams, THERSHOLD);
		delete[] planeParams;

		Point3i currPlaneColor = planeColors[planeColorCount];

		for (int idx = 0; idx < num; idx++) 
		{
			Point3i newColor = Point3i(0, 0, 0);

			if (differences.isInliers.at(idx)) // inlier
			{
				newColor.x = currPlaneColor.x;
				newColor.y = currPlaneColor.y;
				newColor.z = currPlaneColor.z;

				if (colorFlagPush) // First run
				{
					pointsToRemove.push_back(Point3f(0.0, 0.0, 0.0)); // Flagging points to remove with 0: if a point has a color=>remove (add [0,0,0])
				}
				else if (pointsToRemove[idx].x != 0 && pointsToRemove[idx].y != 0 && pointsToRemove[idx].y != 0) // Not first run
				{
					pointsToRemove[idx] = Point3f(0.0, 0.0, 0.0); // If inlier --> Flag as removable
				}
			}
			else // Not inlier
			{
				if (colorFlagPush) // First run
				{
					pointsToRemove.push_back(points[idx]); // Points that do not get removed: newColor=>[0, 0, 0]
				}
				else if (pointsToRemove[idx].x != 0 && pointsToRemove[idx].y != 0 && pointsToRemove[idx].y != 0) // Not first run and point has never been an inlier
				{
					pointsToRemove[idx] = points[idx]; // Load the same point coordinates ==> don't remove
				}
			}

			if (colorFlagPush) // First run
			{
				colorsRANSAC.push_back(newColor);
			}
			else if (!colorFlagPush && colorsRANSAC[idx].x == 0 && colorsRANSAC[idx].y == 0 && colorsRANSAC[idx].z == 0) // If a given point was never colored
			{
				colorsRANSAC[idx] = newColor;
			}
		}

		if (i<num_planes) points_dec = removeInliers(pointsToRemove);
		colorFlagPush = false;
		planeColorCount++;

		//Write a partial pont cloud (for debugging purposes )
		//std::string s = std::to_string(planeColorCount) + argv[2];
		//char const *name = s.c_str
		//WritePLY(name, points_dec, colorsRANSAC);

		cout << "End of fitting of plane " << planeColorCount << endl;
		cout << endl;
	}
	
	vector<string> names = splitString(argv[1], '\\'); // Create a unique name for the model and save it
	string name = getFileName(names) + argv[2];
	char const *name_char = name.c_str();
	name_char = name_char;
	WritePLY(name_char, points, colorsRANSAC);
	cout << "File is ready: " << name_char << endl;
}