#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include<vector>
#include<sstream>

using namespace std;
using namespace cv;

class PlateLocate
{
public:
	PlateLocate();

	int plateLocate(Mat, vector<Mat>&);

	bool verifySizes(RotatedRect mr);

	Mat showResultMat(Mat src, Size rect_size, Point2f center, int index);
};

