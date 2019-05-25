#include "PlateLocate.h"

PlateLocate::PlateLocate()
{
}

int PlateLocate::plateLocate(Mat src, vector<Mat>& resultVec)
{
	Mat src_blur, src_gray;
	Mat grad;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	if (!src.data)
	{
		return -1;
	}

	GaussianBlur(src, src_blur, Size(5, 5),
		0, 0, BORDER_DEFAULT);

	cvtColor(src_blur, src_gray, CV_RGB2GRAY);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 1, abs_grad_y, 0, 0, grad);

	Mat img_threshold;
	threshold(grad, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);

	vector< vector< Point> > contours;
	findContours(img_threshold, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Mat result;

	vector<vector<Point> >::iterator itc = contours.begin();

	vector<RotatedRect> rects;
	int t = 0;
	while (itc != contours.end())
	{
		RotatedRect mr = minAreaRect(Mat(*itc));
		if (!verifySizes(mr))
		{
			itc = contours.erase(itc);
		}
		else
		{
			++itc;
			rects.push_back(mr);
		}
	}

	int k = 1;
	for (int i = 0; i< rects.size(); i++)
	{
		RotatedRect minRect = rects[i];
		if (verifySizes(minRect))
		{
			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			Size rect_size = minRect.size;
			if (r < 1)
			{
				angle = 90 + angle;
				swap(rect_size.width, rect_size.height);
			}
			if (angle - 60 < 0 && angle + 60 > 0)
			{
				Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
				Mat img_rotated;
				warpAffine(src, img_rotated, rotmat, src.size(), CV_INTER_CUBIC);

				Mat resultMat;
				resultMat = showResultMat(img_rotated, rect_size, minRect.center, k++);

				resultVec.push_back(resultMat);
			}
		}
	}
	return 0;
}

bool PlateLocate::verifySizes(RotatedRect mr)
{
	float error = 0.9;
	float aspect = 3.75;
	int min = 44 * 14 * 1;
	int max = 44 * 14 * 24;
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;

	int area = mr.size.height * mr.size.width;
	float r = (float)mr.size.width / (float)mr.size.height;
	if (r < 1)
	{
		r = (float)mr.size.height / (float)mr.size.width;
	}

	if ((area < min || area > max) || (r < rmin || r > rmax))
	{
		return false;
	}
	else
	{
		return true;
	}
}

Mat PlateLocate::showResultMat(Mat src, Size rect_size, Point2f center, int index)
{
	Mat img_crop;
	getRectSubPix(src, rect_size, center, img_crop);

	Mat resultResized;
	resultResized.create(36, 136, CV_8UC3);

	resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);

	return resultResized;
}