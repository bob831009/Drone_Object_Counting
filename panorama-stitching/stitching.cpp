#include <stdio.h>  
#include <string.h>
#include <opencv2/opencv.hpp>  
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/stitching/stitcher.hpp>

using namespace cv;  
using namespace std;

#define IM_START 1
#define IM_NUM 43
int main()  
{
	vector<Mat> vImg;
	Mat rImg;
	vector< vector<Rect> > Transform;

	FILE *fp = fopen("./Stitch_Front/Stitch_info.txt", "w");
	// FILE *fp = fopen("./Stitch_Side/Stitch_info.txt", "w");
	for (int i = IM_START; i <= IM_NUM ; i++){
		char im_name[50];
		if(i < 10){
			sprintf(im_name, "000%d.jpg", i);
		}else{
			sprintf(im_name, "00%d.jpg", i);
		}
		char im_path[100];
		strcpy(im_path, "/auto/extra/b02902015/Drone_Video/Front_Side/");
		// strcpy(im_path, "/auto/extra/b02902015/Drone_Video/Side/");
		strcat(im_path, im_name);
		// ========show each image=========
		// imshow(im_name, imread(im_path));
		// ================================
		vImg.push_back(imread(im_path));
		if(i == IM_START) continue;
		printf("============================\n");
		printf("handling %s image\n", im_name);
		Stitcher stitcher = Stitcher::createDefault(1);
		// stitcher.setWaveCorrection(false);
		stitcher.setWarper(new SphericalWarper());
		Stitcher::Status status = stitcher.stitch(vImg, rImg);
		char output_path[50];
		strcpy(output_path, "/auto/extra/b02902015/panorama-stitching/Stitch_Front/");
		// strcpy(output_path, "/auto/extra/b02902015/panorama-stitching/Stitch_Side/");
		strcat(output_path, im_name);
		int overlap_region = vImg[0].cols + vImg[1].cols - rImg.cols;
		if(status == Stitcher::OK && rImg.rows > 2 && overlap_region > 400){
			imwrite(output_path, rImg);
			printf("%s`s size: %d %d\n", im_name, rImg.rows, rImg.cols);
			printf("%s`s overlap_region: %d\n", im_name, overlap_region);
			fprintf(fp, "%s %d\n", im_name, overlap_region);
		}
		// imshow("Stitching Result",rImg);
		vImg.erase(vImg.begin());
	}
	waitKey(0);
	
	/* =================before code=====================
	Stitcher stitcher = Stitcher::createDefault(1);
	Stitcher::Status status = stitcher.stitch(vImg, rImg);

	if (Stitcher::OK == status) 
  		imshow("Stitching Result",rImg);
  	else
  		printf("Stitching fail.");
	waitKey(0);
	imwrite("output.jpg", rImg);
	*/
}  