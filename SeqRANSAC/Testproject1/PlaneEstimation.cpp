/* Plane Estimators
 *
 * Implemented by Levente Hajder & Daniel Kuknyo
 * hajder@inf.elte.hu
 * daniel.kuknyo@mailbox.org
 * 01-07-2021
 */

#include "PlaneEstimation.h"

Point3f centerOfGravity(vector<Point3f> pts) // Calculate the center of gravity given a set of points in 3D
{
	Point3f cog;
	int num = pts.size();

	for (int i = 0; i < num; i++)
	{
		cog.x += pts.at(i).x;
		cog.y += pts.at(i).y;
		cog.z += pts.at(i).z;
	}

	cog.x /= num;
	cog.y /= num;
	cog.z /= num;

	return cog;
}

float* EstimatePlaneImplicit(vector<Point3f> pts)
{
    int num = pts.size();
	Point3f cog = centerOfGravity(pts);
	Mat Q(num, 4, CV_32F);

    for (int idx = 0; idx < num; idx++) // Form a linear set of equations
	{
        Point3d pt = pts.at(idx);
		Q.at<float>(idx, 0) = pt.x - cog.x;
		Q.at<float>(idx, 1) = pt.y - cog.y;
		Q.at<float>(idx, 2) = pt.z - cog.z;
		Q.at<float>(idx, 3) = 1;
    }

	Mat mtx = Q.t() * Q;

	Mat evals, evecs;
	eigen(mtx, evals, evecs);

    float A = evecs.at<float>(3, 0);
    float B = evecs.at<float>(3, 1);
    float C = evecs.at<float>(3, 2);
	float D = (-A * cog.x - B * cog.y - C * cog.z);
    
    float norm = sqrt(A*A + B*B + C*C);
    
    float* ret = new float[4];
        
    ret[0] = A / norm;
    ret[1] = B / norm;
    ret[2] = C / norm;
    ret[3] = D / norm;

    return ret;
}
    

float* EstimatePlaneRANSAC(vector<Point3f> pts, float threshold, int iterateNum)
{
        int num=pts.size();
        int bestSampleInlierNum=0;
        float bestPlane[4];
        
    for(int iter=0;iter<iterateNum;iter++)
	{
            float rand1=(float)(rand())/RAND_MAX;
            float rand2=(float)(rand())/RAND_MAX;
            float rand3=(float)(rand())/RAND_MAX;
                
            //Generate three different(!) random numbers:
            int index1=(int)(rand1*num);
            int index2=(int)(rand2*num);
            while (index2==index1) {rand2=(float)(rand())/RAND_MAX; index2=(int)(rand2*num);}
            int index3=(int)(rand3*num);
            while ((index3==index1)||(index3==index2)) {rand3=(float)(rand())/RAND_MAX; index3=(int)(rand3*num);}
            
            Point3f pt1=pts.at(index1);
            Point3f pt2=pts.at(index2);
            Point3f pt3=pts.at(index3);
            
            vector<Point3f> minimalSample;
            
            minimalSample.push_back(pt1);
            minimalSample.push_back(pt2);
            minimalSample.push_back(pt3);
            
            float* samplePlane = EstimatePlaneImplicit(minimalSample);
            
            RANSACDiffs sampleResult=PlanePointRANSACDifferences(pts, samplePlane, threshold);
            
            if (sampleResult.inliersNum>bestSampleInlierNum)
			{
                bestSampleInlierNum = sampleResult.inliersNum;
                bestPlane[0] = samplePlane[0];
                bestPlane[1] = samplePlane[1];
                bestPlane[2] = samplePlane[2];
                bestPlane[3] = samplePlane[3];
            }
            delete[] samplePlane;
    }
    
    RANSACDiffs bestResult=PlanePointRANSACDifferences(pts, bestPlane, threshold);
    
    vector<Point3f> inlierPts;
    
    for (int idx=0;idx<num;idx++)
	{
        if (bestResult.isInliers.at(idx))
		{
            inlierPts.push_back(pts.at(idx));
        }
    }
    
    float* finalPlane=EstimatePlaneImplicit(inlierPts);

    return finalPlane;
}

RANSACDiffs PlanePointRANSACDifferences(vector<Point3f> pts, float* plane, float threshold)
{
	int num=pts.size();
        
	float A=plane[0];
	float B=plane[1];
	float C=plane[2];
	float D=plane[3];        

	RANSACDiffs ret;
        
	vector<bool> isInliers;
	vector<float> distances;
        
	int inlierCounter = 0;
	for (int idx = 0; idx < num; idx++)
	{
		Point3f pt = pts.at(idx);
		float diff = fabs(A*pt.x + B*pt.y + C*pt.z + D);
		distances.push_back(diff);
            
		if (diff < threshold)
		{
			isInliers.push_back(true);
			inlierCounter++;
		}
		else
		{
			isInliers.push_back(false);
		}
            
	}
        
	ret.distances=distances;
	ret.isInliers=isInliers;
	ret.inliersNum=inlierCounter;
        
	return ret;
}
