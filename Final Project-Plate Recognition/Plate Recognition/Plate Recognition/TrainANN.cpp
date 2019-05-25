#include "TrainANN.h"



TrainANN::TrainANN()
{
}


TrainANN::~TrainANN()
{
}

void TrainANN::train_ann()
{
	const string fileform = ".lst";
	const string perfileReadPath = "C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\ann_train";

	const int sample_mun_perclass = 50; 
	const int class_mun = 34; 

	const int image_cols = 8;
	const int image_rows = 16;
	string  fileReadName, fileReadPath;
	char temp[256];

	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } }; 
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };

	for (int i = 0; i < class_mun; i++)
	{  
		int j = 0;

		if (i <= 9)  
		{
			sprintf(temp, "%d", i);
		}
		else if (i > 9 && i <= 17)
		{
			sprintf(temp, "%c", i + 55);
		}
		else if (i > 17 && i <= 22)
		{
			sprintf(temp, "%c", i + 56);
		}
		else
		{
			sprintf(temp, "%c", i + 57);
		}
		fileReadPath = perfileReadPath + "\\" + temp + fileform;
		cout << "Process: " << fileReadPath << endl;

		ifstream fin(fileReadPath);
		string img_path;

		while (getline(fin, img_path))
		{
			j++;
			Mat srcImage = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
			Mat resizeImage;
			Mat trainImage;
			Mat result;

			resize(srcImage, resizeImage, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);
			threshold(resizeImage, trainImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			for (int k = 0; k<image_rows*image_cols; ++k)
			{
				trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)trainImage.data[k];
			}
		}
	}

	Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
	cout << "trainingDataMat finished" << endl;

	for (int i = 0; i <= class_mun - 1; ++i)
	{
		for (int j = 0; j <= sample_mun_perclass - 1; ++j)
		{
			for (int k = 0; k < class_mun; ++k)
			{
				if (k == i)
					if (k == 18)
					{
						labels[i*sample_mun_perclass + j][1] = 1;
					}
					else if (k == 24)
					{
						labels[i*sample_mun_perclass + j][0] = 1;
					}
					else
					{
						labels[i*sample_mun_perclass + j][k] = 1;
					}
				else
					labels[i*sample_mun_perclass + j][k] = 0;
			}
		}
	}

	Mat labelsMat(class_mun*sample_mun_perclass, class_mun, CV_32FC1, labels);
	cout << "labelsMat finished" << endl;

	cout << "training start" << endl;
	CvANN_MLP bp;
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.001;
	params.bp_moment_scale = 0.1;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001);  

	Mat layerSizes = (Mat_<int>(1, 5) << image_rows*image_cols, 128, 128, 128, class_mun);
	bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1.0, 1.0);

	cout << "training...." << endl;
	bp.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	bp.save("C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\Model.xml"); //save classifier  
	cout << "training finished" << endl;
	return;
}

char TrainANN::predict(Mat testroi)
{
	cvtColor(testroi, testroi, CV_BGR2GRAY);
	threshold(testroi, testroi, 100, 255, CV_THRESH_BINARY);

	CvANN_MLP bp;
	bp.load("C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\Model.xml");
	const int image_cols = 8;
	const int image_rows = 16;

	Mat test_temp;
	resize(testroi, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);
	threshold(test_temp, test_temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Mat_<float>sampleMat(1, image_rows*image_cols);
	for (int i = 0; i<image_rows*image_cols; ++i)
	{
		sampleMat.at<float>(0, i) = (float)test_temp.at<uchar>(i / 8, i % 8);
	}

	Mat responseMat;
	bp.predict(sampleMat, responseMat);
	Point maxLoc;
	double maxVal = 0;
	minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);
	char temp[256];

	if (maxLoc.x <= 9) 
	{
		sprintf(temp, "%d", maxLoc.x);
	}
	else if (maxLoc.x > 9 && maxLoc.x <= 17)
	{
		sprintf(temp, "%c", maxLoc.x + 55);
	}
	else if (maxLoc.x > 17 && maxLoc.x <= 22)
	{
		sprintf(temp, "%c", maxLoc.x + 56);
	}
	else
	{
		sprintf(temp, "%c", maxLoc.x + 57);
	}
	return temp[0];
}
