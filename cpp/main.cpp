/////////////////////////////////////////////////////////////////////////
// Author:      Zhongze Hu
// Subject:     Circulant Structure of Tracking-by-detection with Kernels
// Algorithm:   ECCV12, Jo~ao F. Henriques, Exploiting the Circulant 
//			    Structure of Tracking-by-detection with Kernels
// Matlab code: http://home.isr.uc.pt/~henriques/circulant/index.html
// Date:        01/13/2015
/////////////////////////////////////////////////////////////////////////



#include "CSK_Tracker.h"
#include <iostream>
#include <fstream>

using namespace std;

void main()
{
	
	CSK_Tracker my_tracker;
	string file_name;
	ifstream infile("input/Dudek/Name.txt");
	//getline(infile,file_name);
	//my_tracker.run_tracker("..\\..\\data\\tiger.avi",Point(16 + 36/2,28 + 36/2),36);
	//my_tracker.run_tracker("..\\..\\data\\boy.avi",Point(374+68/2, 77+68/2),68);
	//my_tracker.run_tracker("..\\..\\CSK\\data\\oldman.avi",Point(186+50/2, 118+50/2),50);

	//VideoCapture capture("input/bike1.avi");
	//Mat frame = imread(file_name);
	////if (!capture.isOpened())
	//if(frame.empty())
	//{
	//	cout << "open video failed!" << endl;
	//	return;
	//}
	//int frame_count = int(capture.get(CV_CAP_PROP_FRAME_COUNT));
	int frame_count = 1490;
	//double rate = capture.get(CV_CAP_PROP_FPS);
	//int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	//int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int width = 320;
	int height = 240;
	Mat frame;
	Mat frame_last;
	Mat alphaf;
	Point pos_first = Point(189,175);
	//int target_sz = 68;
	int target_sz[2] = {176,132};
	namedWindow("haha");
	my_tracker.tracke_one(infile,pos_first,frame_count,target_sz);

}