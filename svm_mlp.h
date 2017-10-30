
#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <time.h>
#include <features.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

//void train_predict_svm( const vector< Mat > & trainData, Mat labels, Mat & testData, Mat & prediction);
void train_predict_mlp( Mat & data,Mat & test_data, Mat & result, string features);
int getPredictedClass(const cv::Mat& predictions);



//void train_predict_svm( const vector< Mat > & trainData, Mat labels, Mat & testData, Mat & prediction)
//{
//    Ptr<SVM> svm = SVM::create();
//
//    svm->setTermCriteria(TermCriteria( TermCriteria::MAX_ITER, 1000, 1e-3 ));
//    svm->setKernel(SVM::LINEAR);
//    svm->setC(0.01); // From paper, soft classifier
//    svm->setType(SVM::C_SVC); // EPS_SVR; // EPSILON_SVR; // may be also NU_SVR; // do regression task
//    svm->train(trainData, ROW_SAMPLE, labels);
//    
//    //svm->save( "my_people_detector.yml" );
//
//    svm->predict(testData,prediction);
//    
//}

void train_predict_mlp( Mat & data,Mat & test_data, Mat & result, string features)
{
    Mat_<float> responses;
    for (int i = 0; i< 10; ++i){
        Mat binary = Mat::zeros(data.rows/10, 10, CV_32FC1);
        binary.col(i) = 1;
        responses.push_back(binary);
    }
    
    //create the neural network
    //Mat_<int> layerSizes(81, 10);
    
    Ptr<ANN_MLP> network = ANN_MLP::create();
    //network->setLayerSizes(layerSizes);
    vector<int> layerSizes;
    if (features == "pixel") {
        layerSizes = { test_data.cols, test_data.cols / 2,
            responses.cols };
    }
    else
        layerSizes = { test_data.cols, test_data.cols / 2, test_data.cols/4,
            responses.cols };
    network->setLayerSizes(layerSizes);
    network->setActivationFunction(ANN_MLP::SIGMOID_SYM);
    network->setTrainMethod(ANN_MLP::BACKPROP);
    Ptr<TrainData> trainData = TrainData::create(data, ROW_SAMPLE, responses);
    
    network->train(trainData);
    
    
    network->predict(test_data,result);
}

int getPredictedClass(const cv::Mat& predictions) //ADD REFERENCE
{
    float maxPrediction = predictions.at<float>(0);
    float maxPredictionIndex = 0;
    const float* ptrPredictions = predictions.ptr<float>(0);
    for (int i = 0; i < predictions.cols; i++)
    {
        float prediction = *ptrPredictions++;
        if (prediction > maxPrediction)
        {
            maxPrediction = prediction;
            maxPredictionIndex = i;
        }
    }
    return maxPredictionIndex;
}

