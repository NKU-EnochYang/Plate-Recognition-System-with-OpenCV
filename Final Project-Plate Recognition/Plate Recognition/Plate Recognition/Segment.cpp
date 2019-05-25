#include "Segment.h"



Segment::Segment()
{
}


Segment::~Segment()
{
}

int Segment::getColSum(Mat & bimg, int col)
{
	int height = bimg.rows;
	int sum = 0;
	for (int i = 1; i < height; i++)
	{
		sum += bimg.at<uchar>(i, col);
	}
	return sum;
}

int Segment::cutLeft(Mat & src, int Tsum, int right)
{
	int left;
	left = 0;

	int i;
	for (i = 0; i < src.cols; i++)
	{
		int colValue = getColSum(src, i);
		if (colValue> Tsum)
		{
			left = i;
			break;
		}
	}
	int roiWidth = src.cols / 7;
	for (; i < src.cols; i++)
	{
		int colValue = getColSum(src, i);
		if (colValue < Tsum)
		{
			right = i;
			if ((right - left) < (src.cols / 7))
				continue;
			else
			{
				roiWidth = right - left;
				break;
			}

		}
	}
	return roiWidth;
}

int Segment::getOne(Mat & inimg)
{
	Mat gimg, histimg;
	cvtColor(inimg, gimg, CV_BGR2GRAY);
	equalizeHist(gimg, histimg);

	threshold(gimg, gimg, 100, 255, CV_THRESH_BINARY);

	int psum = 0;
	for (int i = 0; i < gimg.cols; i++)
	{
		psum += getColSum(gimg, i);
	}

	int Tsum = 0.6*(psum / gimg.cols);
	int roiWid = cutLeft(gimg, Tsum, 0);
	if (roiWid > 136 / 7)
		roiWid = 136 / 7;
	return roiWid;
}

vector<Mat> Segment::seg(Mat src)
{
	vector<Mat> seg;
	int idx = getOne(src);
	Mat img = src(Range(0, src.rows), Range(idx, src.cols));
	Mat temp;
	seg = seg_img(img);
	return seg;
}

vector<Mat> Segment::seg_img(Mat imgSrc)
{
	Mat img_gray;
	cvtColor(imgSrc, img_gray, CV_BGR2GRAY);
	threshold(img_gray, img_gray, 100, 255, CV_THRESH_BINARY);
	
	vector<vector<cv::Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	vector<Mat> res;
	vector<Rect> rec;
	for (int i = 0; i<contours.size(); i++)
	{
		Rect rect = boundingRect(Mat(contours[i]));
		if (verifySizes(imgSrc(rect)))
		{
			res.push_back(imgSrc(rect));
			rec.push_back(rect);
		}
	}
	for (int i = 0; i < rec.size() - 1; i++)
	{
		for (int j = 0; j < rec.size() - 1 - i; j++)
		{
			if (rec[j].x > rec[j + 1].x)
			{
				swap(rec[j], rec[j + 1]);
				swap(res[j], res[j + 1]);
			}
		}
	}
	if (res.size() > 6)
		res.pop_back();
	return res;
}

bool Segment::verifySizes(Mat r) {
	cvtColor(r, r, CV_BGR2GRAY);

	float aspect = 45.0f / 90.0f;
	float charAspect = (float)r.cols / (float)r.rows;
	float error = 0.7f;
	float minHeight = 10.f;
	float maxHeight = 35.f;

	float minAspect = 0.05f;
	float maxAspect = aspect + aspect * error;

	int area = cv::countNonZero(r);

	int bbArea = r.cols * r.rows;

	int percPixels = area / bbArea;

	if (percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect &&
		r.rows >= minHeight && r.rows < maxHeight)
		return true;
	else
		return false;
}