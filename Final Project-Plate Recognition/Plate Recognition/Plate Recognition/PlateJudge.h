#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2/ml/ml.hpp>
#include<vector>
#include<sstream>

using namespace std;
using namespace cv;

class PlateJudge
{
public:
	PlateJudge();
	~PlateJudge();
	int plateJudge(const vector<Mat>& inVec, vector<Mat>& resultVec);
private:
	CvSVM svm;
};

