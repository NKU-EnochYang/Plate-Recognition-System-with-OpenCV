#include "PlateJudge.h"

PlateJudge::PlateJudge()
{
	svm.load("C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\SVM_HOG.xml");
}


PlateJudge::~PlateJudge()
{
}


int PlateJudge::plateJudge(const vector<Mat>& inVec, vector<Mat>& resultVec)
{
	HOGDescriptor hog(Size(136, 36), Size(12, 12), Size(4, 4), Size(4, 4), 9);
	int num = inVec.size();
	for (int j = 0; j < num; j++)
	{
		Mat inMat = inVec[j];
		vector<float> desc;
		hog.compute(inMat, desc, Size(4, 4));
		CvMat *feat = cvCreateMat(1, desc.size(), CV_32FC1);
		int n = 0;
		for (vector<float>::iterator iter = desc.begin(); iter != desc.end(); iter++)
		{
			cvmSet(feat, 0, n, *iter);
			n++;
		}
		int response = (int)svm.predict(feat);
		if (response == 1)
		{
			resultVec.push_back(inMat);
		}
	}
	return 0;
}
