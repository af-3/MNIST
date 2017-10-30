//
//  main.cpp
//  Lab_2_MNIST
//
//  Created by Alice Fockele on 4/24/17.
//  Copyright © 2017 Alice Fockele. All rights reserved.
//


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>


#include <iostream>
#include <fstream>
#include <string> 

#include <ctype.h>
#include <svm_mlp.h>
#include <features.h>
#include <error_reporting.h>

using namespace cv;
using namespace std;
using namespace cv::ml;

void writeCSV(string filename, Mat m); //https://gist.github.com/zhou-chao/7a7de79de47c652196f1

int main(int argc, const char * argv[]) {
    string user_features;
    cout << "features: " << endl;
    cin >> user_features;
    string user_machine;
    cout << "machine: " << endl;
    cin >> user_machine;
    
//READ IN IMAGE
    Mat image_in;
    const string image_name = "digits.png";
    image_in = imread(image_name,0);
    namedWindow("image in", WINDOW_AUTOSIZE);
    imshow("image in", image_in);
//SPLIT IMAGE INTO TRAIN AND TEST
    float train_percent = 0.7; //indicates 70 percent of input image
    Mat image_train = image_in.colRange(Range(0,image_in.cols*train_percent));
    Mat image_test = image_in.colRange(Range(image_in.cols*train_percent, image_in.cols));

//ORGANIZE TRAINING IMAGES
    vector<Mat> image_lst;
    for (int i = 0; i<image_train.rows; i = i+20) {
        for (int j = 0; j<image_train.cols; j = j+20) {
            image_lst.push_back(image_train(Rect(j,i,20,20)));
        }
    }
    
//ORGANIZE TESTING IMAGES
    vector<Mat> image_test_lst;
    for (int i = 0; i<image_test.rows; i = i+20) {
        for (int j = 0; j<image_test.cols; j = j+20) {
            image_test_lst.push_back(image_test(Rect(j,i,20,20)));
        }
    }
    //cout<<image_test_lst[0].type()<<endl;
    
    
//TRAINING LABELS
    Mat labels_train;
    for (int k = 0; k < 10; k++) {
        Mat binary = Mat::ones(500*train_percent,1, CV_32SC1)*k;
        labels_train.push_back(binary);
    }
    writeCSV("labels.csv", labels_train);
    //cout <<"Labels zero to one transition" << endl << labels_train.rowRange(Range(500*train_percent-5,500*train_percent+5)) << endl;
    
//TEST LABELS
    Mat labels_test;
    for (int k = 0; k < 10; k++) {
        Mat binary2 = Mat::ones(500*(1-train_percent),1, CV_32SC1)*k;
        labels_test.push_back(binary2);
    }
    
//CALC TRAIN AND TEST DATA
    Mat train_data;
    Mat test_data;
    
    if (user_features == "pixel") {
        //COMPUTE PIXEL METRICS
        compute_pixel(image_lst, train_data);
        compute_pixel(image_test_lst, test_data);
    }

    else if (user_features == "hog"){
        //COMPUTE HOG METRICS
        compute_hog(image_lst, train_data); //from opencv cpp samples folder
        compute_hog(image_test_lst, test_data); //from opencv cpp samples folder

    }
    
//TRAIN AND PREDICT
    Mat prediction;
    //vector<int> error_vec;
    string filename = "predict_" + user_machine + "_" + user_features ;
    cout<< "filename: " << filename << endl;
    
    if (user_machine == "svm") {
        //Mat prediction(test_data.rows,1,CV_32FC1);
        Ptr<SVM> svm = SVM::create();
        /* Default values to train SVM */
        svm->setCoef0(0.0);
        svm->setDegree(3);
        svm->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-3 ));
        svm->setGamma(0.5);
        svm->setKernel(SVM::RBF);
        svm->setNu(0.5);
        svm->setP(1); // for EPSILON_SVR, epsilon in loss function?
        svm->setC(1); // From paper, soft classifier
        svm->setType(SVM::C_SVC); // EPS_SVR; // EPSILON_SVR; // may be also NU_SVR; // do regression task
        
        svm->train(train_data, ROW_SAMPLE, labels_train);
        
        svm->save( filename + "_detector.yml" );
        svm->predict(test_data,prediction);
        
        prediction.convertTo(prediction, CV_32SC1);
        int error = get_error_rate(prediction, labels_test);
        cout << "Error: " << error << endl;
        
        //Display Image
        vector<int> error_vec;
        //Mat prediction_int;
        //prediction.convertTo(prediction_int, CV_32SC1);
        absdiff(prediction, labels_test, error_vec);
        
        Mat displayImage = image_test.clone();
        Mat displayImageText = Mat::zeros(displayImage.rows, displayImage.cols, CV_32FC1);
        Mat displayImageMask = Mat::zeros(displayImage.rows, displayImage.cols, CV_32FC1);
        
        for (int i = 0; i<image_test_lst.size(); ++i) {
            if (error_vec[i] != 0) {
                int r = i/30; //(image_test.rows/20);
                int c = i%30 - 1; //(image_test.rows/20);
                circle(displayImage, Point(c*20+10,r*20+10), 10, Scalar(255,255,255));
                //cout << "index: " << i << endl;
                putText(displayImageText, to_string(prediction.at<int>(0,i)), Point(c*20-10,r*20+30), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, Scalar(255,255,255),1);
                circle(displayImageMask, Point(c*20+10,r*20+10) , 10, Scalar(255,255,255), -1);
            }
            continue;
        }
        displayImage.convertTo(displayImage, CV_32FC1);
        Mat comp = (displayImage & displayImageMask);
        Mat comp_predict = (comp | displayImageText);
        imshow("missed digits",displayImage);
        imshow("masked?", comp_predict);
        imwrite(filename + to_string(error) + ".png", displayImage);
        imwrite(filename + "_maksed.png",comp_predict);
        
        writeCSV(filename + to_string(error) + ".csv", prediction);
    }
    
    else if (user_machine == "mlp"){

        train_predict_mlp(train_data, test_data, prediction, user_features);
        Mat confusionMatrix;
        for (int i = 0; i < prediction.rows; i++)
        {
            int predictedClass = getPredictedClass(prediction.row(i));
            //int expectedClass = testOutputExpected.at(i);
            confusionMatrix.push_back(predictedClass);
        }
        //writeCSV("result_classes_mlp.csv", confusionMatrix);
        writeCSV(filename + ".csv", confusionMatrix);
        confusionMatrix.convertTo(confusionMatrix, CV_32SC1);
        int error = get_error_rate(confusionMatrix, labels_test);
        cout << "Error: " << error << endl;
        
        //Display Image
        vector<int> error_vec;
        absdiff(confusionMatrix, labels_test, error_vec);
        
        Mat displayImage = image_test.clone();
        Mat displayImageText = Mat::zeros(displayImage.rows, displayImage.cols, CV_32FC1);
        Mat displayImageMask = Mat::zeros(displayImage.rows, displayImage.cols, CV_32FC1);

        for (int i = 0; i<image_test_lst.size(); ++i) {
            if (error_vec[i] != 0) {
                int r = i/30; //(image_test.rows/20);
                int c = i%30 - 1; //(image_test.rows/20);
                circle(displayImage, Point(c*20+10,r*20+10), 10, Scalar(255,255,255));
                //cout << "index: " << i << endl;
                putText(displayImageText, to_string(confusionMatrix.at<int>(0,i)), Point(c*20-10,r*20+30), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, Scalar(255,255,255),1);
                circle(displayImageMask, Point(c*20+10,r*20+10) , 10, Scalar(255,255,255), -1);
            }
            continue;
        }
        displayImage.convertTo(displayImage, CV_32FC1);
        Mat comp = (displayImage & displayImageMask);
        Mat comp_predict = (comp | displayImageText);
        imshow("missed digits",displayImage);
        imshow("masked?", comp_predict);
        imwrite(filename + to_string(error) + ".png", displayImage);
        imwrite(filename + "_maksed.png",comp_predict);
    }
    

//    vector<Mat> error_img_lst;
//    for (int i =0; i<image_test_lst.size(); ++i) {
//        if (error[i] != 0) {
//            error_img_lst.push_back(image_test_lst[i]);
//        }
//    }
    

    
//    Mat displayImage = Mat::zeros(20,error_img_lst.size()*20, CV_32FC1);
//    for (int i = 0; i < error_img_lst.size(); i++) {
//            Mat test(20,20,CV_32FC1);
//            test = error_img_lst[i].clone();
//            //Mat predict_value(20,20,CV_32FC1);
//            error_img_lst[i].clone().copyTo(displayImage(Rect(i*20,0,20,20)));
//    }
//    imshow("result error",displayImage);

    
    
    waitKey();
    destroyAllWindows();

    
    return 0;
}

void writeCSV(string filename, Mat m)
{
    ofstream myfile;
    myfile.open(filename.c_str());
    myfile<< cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    myfile.close();
}
