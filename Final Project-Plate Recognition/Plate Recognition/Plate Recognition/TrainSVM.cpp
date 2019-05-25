#include "TrainSVM.h"

TrainSVM::TrainSVM()
{
}


TrainSVM::~TrainSVM()
{
}

void TrainSVM::train_svm()
{
	HOGDescriptor hog(Size(136, 36), Size(12, 12), Size(4, 4), Size(4, 4), 9);

	int descriptorDim = 0;

	ofstream foutVector("Dim.txt");
	string img_path;

	const string POS_IMAGE = "C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\svm_train\\has\\train_pos.lst";
	const string NEG_IMAGE = "C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\svm_train\\no\\train_neg.lst";

	int pos_num = 1400;
	int neg_num = 2174;

	ifstream fin_pos(POS_IMAGE);
	ifstream fin_neg(NEG_IMAGE);

	Mat trainFeatureMat;
	Mat trainLabelMat;

	for (int i = 0; i<pos_num && getline(fin_pos, img_path); i++)
	{
		cout << "Process: " << img_path << endl;
		Mat img = imread(img_path, 1); 
		vector<float> descriptors;
		hog.compute(img, descriptors, Size(4, 4));
		if (i == 0)
		{
			descriptorDim = descriptors.size();
			foutVector << descriptorDim << endl;
			trainFeatureMat =
				Mat::zeros(pos_num + neg_num, descriptorDim, CV_32FC1);
			trainLabelMat = Mat::zeros(pos_num + neg_num, 1, CV_32FC1); \
		}
		for (int j = 0; j<descriptorDim; j++)
		{
			trainFeatureMat.at<float>(i, j) = descriptors[j];
		}

		trainLabelMat.at<float>(i, 0) = 1; 

	} 
	fin_pos.close();


	for (int i = 0; i<neg_num && getline(fin_neg, img_path); i++)
	{
		cout << "Process: " << img_path << endl;
		Mat img = imread(img_path, 1); 
		vector<float> descriptors;
		hog.compute(img, descriptors, Size(4, 4)); 
													
		for (int j = 0; j<descriptorDim; j++)
		{
			trainFeatureMat.at<float>(i + pos_num, j) = descriptors[j];
		}
		trainLabelMat.at<float>(i + pos_num, 0) = -1;     //positive
	}
	fin_neg.close();

	foutVector.close();
	cout << "Feature Extraction Done" << endl;

	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, 1E-7);
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);

	cout << "Start Training....." << endl;
	double t = (double)getTickCount();
	my_svm.train(trainFeatureMat, trainLabelMat, Mat(), Mat(), param);
	t = (double)getTickCount() - t;
	cout << "Training finished" << endl;
	my_svm.save("C:\\Users\\Seth\\Desktop\\Computer Vision\\Plate Recognition\\SVM_HOG.xml");
	printf("train time = %gms\n", t*1000. / cv::getTickFrequency());
}