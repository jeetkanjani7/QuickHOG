#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/thread/thread.hpp>
#include <fltk/run.h>

#include "HOG/HOGEngine.h"
#include "HOG/HOGImage.h"

#include "Utils/ImageWindow.h"
#include "Utils/Timer.h"

#include "Others/persondetectorwt.tcc"

#define NMS_ARB 128;

using namespace HOG; 
using namespace cv;




float IOUcalc(HOGResult b1, HOGResult b2)
{
	float ai = (float)(b1.width + 1)*(b1.height + 1);
	float aj = (float)(b2.width + 1)*(b2.height + 1);
	float x_inter, x2_inter, y_inter, y2_inter;

	x_inter = max(b1.x,b1.x);
	y_inter = max(b1.y,b2.y);
	
	x2_inter = min((b1.x + b1.width),(b2.x + b2.width));
	y2_inter = min((b1.y + b1.height),(b2.y + b2.height));
	
	float w = (float)max((float)0, x2_inter - x_inter);  
	float h = (float)max((float)0, y2_inter - y_inter);  
	
	float inter = ((w*h)/(ai + aj - w*h));
	return inter;
}


void NMScalc(bool *res, HOGResult *b, int count)
{

    float theta = 0.15;

    for(int i=0; i<count; i++)
    {
    	res[i] = true;
    }

    for(int i=0; i<count; i++)
    {
    	for(int j=0; j<count; j++)
    	{
    		if(b[i].score > b[j].score)
	 	{	 		
	 		if(IOUcalc(b[i],b[j]) > theta)
	 		{ 
	 			res[j] = false; 
	 		}
		}
    	}	
    }  
}


Mat & DrawBoxes(Mat &im,HOGResult *boxResults, int count)
{
	printf("\n Boxes: ");
	for (int i=0; i<count; i++)
	{
		printf("%1.5f %1.5f %4d %4d %4d %4d %4d %4d\n",
		boxResults[i].scale,
		boxResults[i].score,
		boxResults[i].origX,
		boxResults[i].origY,
		boxResults[i].x,
		boxResults[i].y,
		boxResults[i].width,
		boxResults[i].height);
		
		Point orig = Point(boxResults[i].x,boxResults[i].y);

		Point end = orig + Point(boxResults[i].width ,boxResults[i].height);
	
		rectangle(im, orig, end, Scalar(0,0,0,255));	
	}

	//line(im, textOrg + Point(0, thickness),
     //textOrg + Point(textSize.width, thickness),
     //Scalar(0, 0, 255));

	return im;
}


Mat & DrawNms(Mat &im,HOGResult *nmsResults, int count)
{

	printf("\nOPtimum::");
	for (int i=0; i<count; i++)
	{
		printf("%1.5f %1.5f %4d %4d %4d %4d %4d %4d\n",
		nmsResults[i].scale,
		nmsResults[i].score,
		nmsResults[i].origX,
		nmsResults[i].origY,
		nmsResults[i].x,
		nmsResults[i].y,
		nmsResults[i].width,
		nmsResults[i].height);

		Point orig = Point(nmsResults[i].x,nmsResults[i].y);

		Point end = orig + Point(nmsResults[i].width ,nmsResults[i].height);

		rectangle(im, orig, end, Scalar(0,0,255,255)); 
			
	}

	
	

	return im;
}


HOGEngine *doStuffHere(HOGImage *image, HOGImage *imageCUDA)
{
	HOGEngine *Engine = HOGEngine::Instance();
	Engine->InitializeHOG(image->width, image->height,
			PERSON_LINEAR_BIAS, PERSON_WEIGHT_VEC, PERSON_WEIGHT_VEC_LENGTH);

	Timer t;
	t.restart();
	Engine->BeginProcess(image);
	
	Engine->EndProcess();
	
	t.stop();
	
	t.check("Processing time");
	
	printf("Found %d positive results.\n", Engine->formattedResultsCount);
	printf("\n %f %f %d %d %d %d %d %d\n",
			 Engine->formattedResults[0].scale, Engine->formattedResults[0].score,
			 Engine->formattedResults[0].width,  Engine->formattedResults[0].height,
			 Engine->formattedResults[0].x,  Engine->formattedResults[0].y,
			 Engine->formattedResults[0].origX,  Engine->formattedResults[0].origY);

	Engine->GetImage(imageCUDA, HOGEngine::IMAGE_ROI);
	
	printf("Drawn %d positive results.\n", Engine->nmsResultsCount);
		
	return Engine;

}

int main(void)
{

	Mat temp = imread("/home/jeetkanjani7/pedestrian_imgs/pedestrians.jpg",1);

	Mat im = Mat(temp.size(), CV_MAKE_TYPE(temp.type(), 4));
	cvtColor(temp, im, CV_BGR2BGRA, 4);
	
	Mat Oxsight;
	im.copyTo(Oxsight);

	HOGImage *image = new HOGImage(im.cols,im.rows,im.data);
	
	
	HOGImage *imageCUDA = new HOGImage(im.cols,im.rows);

	HOGEngine *res_instance = doStuffHere(image, imageCUDA);

	HOGResult *box_results = res_instance->formattedResults;
	HOGResult *nms_res = res_instance->nmsResults;
	bool *oxsight_res = res_instance->oxsight_results;
	HOGResult *Ox_windows = res_instance -> oxsight_windows;
	im = DrawBoxes(im, box_results, res_instance->formattedResultsCount);
	im = DrawNms(im, nms_res, res_instance->nmsResultsCount);
	
	//int count =  res_instance->formattedResultsCount;
	bool *res = oxsight_res;

	//NMScalc( res, box_results, count);
	
	

	for(int i =0; i< 128; i++)
	{
		printf("\nres: %d",res[i]);
			//std::ostringstream ss;
			//ss <<b[i].s;
			//string text(ss.str());
			//Size textSize = getTextSize(text, fontFace,fontScale, thickness, &baseline);
		if(res[i])
		{
			printf("Results= %f--%d--%d",Ox_windows[i].score,Ox_windows[i].x,Ox_windows[i].y);
			rectangle(Oxsight,Point(Ox_windows[i].x,Ox_windows[i].y),Point(Ox_windows[i].x + Ox_windows[i].width,Ox_windows[i].y + Ox_windows[i].height),Scalar(255,0,0),0.5,8,0);
			//line(temp, Point(box_results[i].x,box_results[i].y),Point(box_results[i].x + box_results[i].width,box_results[i].y + box_results[i].height), Scalar(255, 0, 0));
			//putText(temp, text,  Point(box_results[i].x,box_results[i].y), fontFace, fontScale,Scalar(255,0,0), thickness, 8);

		}
		else
		{
			rectangle(Oxsight,Point(Ox_windows[i].x,Ox_windows[i].y),Point(Ox_windows[i].x + Ox_windows[i].width,Ox_windows[i].y + Ox_windows[i].height),Scalar(0,0,0),0.5,8,0);
			printf("Results= %f--%d--%d",Ox_windows[i].score,Ox_windows[i].x,Ox_windows[i].y);
			//line(temp, Point(box_results[i].x,box_results[i].y),Point(box_results[i].x + box_results[i].width,box_results[i].y + box_results[i].height), Scalar(255, 0, 0));
			//putText(temp, text,  Point(box_results[i].x,box_results[i].y), fontFace, fontScale,Scalar(0,0,255), thickness, 8);
		}
		
	}

	

	imshow("nms_bool",Oxsight);


	
	imshow("cuda_img",im);
	waitKey(0);
	res_instance->FinalizeHOG();
	
	
	
	delete image;
	delete imageCUDA;

	return 0;
}
