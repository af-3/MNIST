//
//  mlp.h
//  Lab_2_MNIST
//
//  Created by Alice Fockele on 4/26/17.
//  Copyright © 2017 Alice Fockele. All rights reserved.
//

#ifndef features_h
#define features_h

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;


void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData );
void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst);
void compute_pixel(const vector< Mat > & img_lst, vector< Mat > & count_lst);



void compute_pixel(const vector< Mat > & img_lst,  cv::Mat& trainData ){
    vector< Mat > count_lst;
    std::vector<float> pixels;
    for (int i = 0; i<img_lst.size();i++) {
        float value = countNonZero(img_lst[i]);
        pixels.clear();
        pixels.push_back(value);
        Scalar avg;
        Scalar stdev;
        meanStdDev(img_lst[i], avg, stdev);
        //Scalar avg = mean(img_lst[i]);
        pixels.push_back(avg[0]);
        pixels.push_back(stdev[0]);
        pixels.push_back(0);
        
        count_lst.push_back(Mat( pixels ).clone() );
    }
    //trainData.convertTo(trainData, CV_32FC1);
    convert_to_ml( count_lst, trainData );
    
}


void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1 );
    vector< Mat >::const_iterator itr = train_samples.begin();
    vector< Mat >::const_iterator end = train_samples.end();
    for( int i = 0 ; itr != end ; ++itr, ++i )
    {
        CV_Assert( itr->cols == 1 ||
                  itr->rows == 1 );
        if( itr->cols == 1 )
        {
            transpose( *(itr), tmp );
            tmp.copyTo( trainData.row( i ) );
        }
        else if( itr->rows == 1 )
        {
            itr->copyTo( trainData.row( i ) );
        }
    }
}


void compute_hog( const vector< Mat > & img_lst, cv::Mat& trainData )
{
    vector<Mat> gradient_lst;
    HOGDescriptor hog( //settings from: https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
                      Size(20,20), //winSize
                      Size(10,10), //blocksize
                      Size(5,5), //blockStride,
                      Size(10,10), //cellSize,
                      9, //nbins,
                      1, //derivAper,
                      -1, //winSigma,
                      0, //histogramNormType,
                      0.2, //L2HysThresh,
                      1,//gammal correction,
                      64,//nlevels=64
                      1);//Use signed gradients
    
    hog.winSize = img_lst[0].size();
    
    vector< Point > location;
    vector< float > descriptors;
    
    location.push_back(Point(img_lst[0].cols/2,img_lst[0].rows/2));
    hog.winSize = img_lst[0].size();
    
    for (int i = 0; i<img_lst.size(); i++)
    {
        
        hog.compute( img_lst[i], descriptors);
        gradient_lst.push_back( Mat( descriptors ).clone() );
    }
    convert_to_ml( gradient_lst, trainData );
}



#endif /* features_h */
