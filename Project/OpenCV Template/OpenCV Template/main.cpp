#include "opencv2\opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	//Preprocessing
	Mat testImageGray = imread("pos00005.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	testImageGray.convertTo(testImageGray, CV_32F, 1/255.0);

	//calculate gradient x & y
	Mat gx, gy;
	Sobel(testImageGray, gx, CV_32F, 1, 0, 1);
	Sobel(testImageGray, gy, CV_32F, 0, 1, 1);

	//calculate gradient magnitude and direction (degrees)
	Mat mag, angle;
	Mat GradientXimage, GradientYimage, MagnitudeImage;
	cartToPolar(gx, gy, mag, angle, 1);

	//absolute gradient x
	addWeighted(gx, 0.5, 0, 0.5, 0, GradientXimage);

	//absolute gradient y
	addWeighted(gy, 0.5, 0, 0.5, 0, GradientYimage);

	//absolute magnitude
	addWeighted(mag, 0.5, 0, 0.5, 0, MagnitudeImage);

	//Display image and values
	namedWindow("pedestrian", CV_WINDOW_KEEPRATIO);
	imshow("pedestrian", testImageGray);
	namedWindow("Gradient X Image", CV_WINDOW_KEEPRATIO);
	imshow("Gradient X Image", GradientXimage);
	namedWindow("Gradient Y Image", CV_WINDOW_KEEPRATIO);
	imshow("Gradient Y Image", GradientYimage);
	namedWindow("Magnitude Image", CV_WINDOW_KEEPRATIO);
	imshow("Magnitude Image", MagnitudeImage);
	cout << "Gradient X: " << gx << endl;
	cout << "Gradient Y: " << gy << endl;
	cout << "Maginutde Gradient: " << mag << endl;
	cout << "Angle Gradient: " << angle << endl;

	waitKey();

	return 0;
}