# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2

# define basic parameter
IM_NUM = 17;
result = [];
stitcher = Stitcher()

for i in range(IM_NUM, 14, -1):
	if(i < 10): 
		im_name = "000"+str(i)+".jpg";
	else: 
		im_name = "00"+str(i)+".jpg";

	im_path = "/auto/extra/b02902015/py-faster-rcnn/video_image/Compress/" + im_name;
	image = cv2.imread(im_path);
	cv2.imshow(im_name ,image);
	if(i == IM_NUM):
		result = image;
		continue;
	result = stitcher.stitch([result, image], showMatches=True, diraction=0);

cv2.imshow("Result", result)
cv2.waitKey(0)