#include "PlateLocate.h"
#include"PlateJudge.h"
#include"TrainSVM.h"
#include"Segment.h"
#include"TrainANN.h"
#include<iostream>

using namespace std;

void train_svm();
void train_ann();
void test(Mat img, int cnt);

int main()
{
	Mat img1 = imread("C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\test1.jpg");
	test(img1, 1);
	resize(img1, img1, Size(img1.cols / 2, img1.rows / 2));
	imshow("img1_src", img1);

	Mat img2 = imread("C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\test1.jpg");
	test(img2, 2);
	resize(img2, img2, Size(img2.cols / 2, img2.rows / 2));
	imshow("img2_src", img2);

	Mat img3 = imread("C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\test2.jpg");
	test(img3, 3);
	resize(img3, img3, Size(img3.cols / 2, img3.rows / 2));
	imshow("img3_src", img3);

	waitKey(0);
	return 0;
}

void train_svm()
{
	TrainSVM trainer;
	trainer.train_svm();
}

void train_ann()
{
	TrainANN ann;
	ann.train_ann();
}

void test(Mat img, int cnt)
{
	string num = to_string(cnt);
	num = "plate" + num;
	vector<Mat> res;
	PlateLocate locator;
	locator.plateLocate(img, res);
	PlateJudge judger;
	vector<Mat> n_res;
	judger.plateJudge(res, n_res);
	Segment segmentor;
	TrainANN ann;
	for (int i = 0; i < n_res.size(); i++)
	{
		Mat temp = n_res[i];
		imshow(num, temp);
		vector<Mat> roi = segmentor.seg(temp);
		char ch;
		cout << "img" << num << ": ";
		for (int j = 0; j < roi.size(); j++)
		{
			string na = to_string(j);
			imshow(na, roi[j]);
			ch = ann.predict(roi[j]);
			cout << ch;
		}
	}
	cout << endl;
}