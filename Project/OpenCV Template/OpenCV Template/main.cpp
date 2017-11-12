#include "opencv2\opencv.hpp"
#include <iostream>
#include "opencv2\objdetect.hpp"
#include <vector>
#include <stdio.h>
#include <stdint.h>

using namespace cv;
using namespace std;

Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);

int main()
{
	//Preprocessing
	Mat testImageGray = imread("pos00005.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	testImageGray.convertTo(testImageGray, CV_32F, 1/255.0);
	resize(testImageGray, testImageGray, Size(18, 36));

	//calculate gradient x & y
	Mat gx, gy;
	Sobel(testImageGray, gx, CV_32F, 1, 0, 1);
	Sobel(testImageGray, gy, CV_32F, 0, 1, 1);

	//calculate gradient magnitude and direction (degrees)
	Mat mag, angle;
	Mat GradientXimage, GradientYimage, MagnitudeImage, AngleImage, gradientImage;
	cartToPolar(gx, gy, mag, angle, 1);

	//absolute gradient x
	addWeighted(gx, 0.5, 0, 0.5, 0, GradientXimage);

	//absolute gradient y
	addWeighted(gy, 0.5, 0, 0.5, 0, GradientYimage);

	//absolute magnitude
	addWeighted(mag, 0.5, 0, 0.5, 0, MagnitudeImage);

	//absolute angle
	//addWeighted(angle, 0.5, 0, 0.5, 0, AngleImage);

	addWeighted(mag, 0.5, angle, 0.5, 0, gradientImage);

	//extract features
	vector<float> DescriptorValues;
	vector<Point> Locations;
	HOGDescriptor d(Size(18,36), Size(6,6), Size(3,3), Size(3,3),9);
	d.compute(gradientImage, DescriptorValues, Size(0, 0), Size(0, 0), Locations);
	cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
	cout << "img dimensions: " << gradientImage.cols << " width x " << gradientImage.rows << "height" << endl;
	cout << "Found " << DescriptorValues.size() << " descriptor values" << endl;
	cout << "Nr of locations specified : " << Locations.size() << endl;

	Mat visualDescriptor = get_hogdescriptor_visu(testImageGray, DescriptorValues, Size(18,36));

	//Display image and values
	namedWindow("pedestrian", CV_WINDOW_KEEPRATIO);
	imshow("pedestrian", testImageGray);
	namedWindow("Gradient X Image", CV_WINDOW_KEEPRATIO);
	imshow("Gradient X Image", GradientXimage);
	namedWindow("Gradient Y Image", CV_WINDOW_KEEPRATIO);
	imshow("Gradient Y Image", GradientYimage);
	namedWindow("Magnitude Image", CV_WINDOW_KEEPRATIO);
	imshow("Magnitude Image", MagnitudeImage);
	//namedWindow("Angle Image", CV_WINDOW_KEEPRATIO);
	//imshow("Angle Image", AngleImage);
	namedWindow("Gradient Image", CV_WINDOW_KEEPRATIO);
	imshow("Gradient Image", gradientImage);
	namedWindow("Visual Descriptor", CV_WINDOW_KEEPRATIO);
	imshow("Visual Descriptor", visualDescriptor);
	/*cout << "Gradient X: " << gx << endl;
	cout << "Gradient Y: " << gy << endl;
	cout << "Maginutde Gradient: " << mag << endl;
	cout << "Angle Gradient: " << angle << endl;*/

	//save  feature descriptor to arff file
	/*ofstream descriptor;
	descriptor.open("postive1.arff");
	descriptor << DescriptorValues.data();
	descriptor.close();*/



	waitKey();

	return 0;
}

Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu