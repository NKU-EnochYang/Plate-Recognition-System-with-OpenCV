#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include<vector>
#include<sstream>

using namespace std;
using namespace cv;

class Segment
{
public:
	Segment();
	~Segment();
	int getColSum(Mat &bimg, int col);
	int cutLeft(Mat &src, int Tsum, int right);
	int getOne(Mat &inimg);
	vector<Mat> seg(Mat img);
	vector<Mat> seg_img(Mat imgSrc);
	bool verifySizes(Mat r);
};

