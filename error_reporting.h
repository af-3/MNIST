//
//  error_reporting.h
//  Lab_2_MNIST
//
//  Created by Alice Fockele on 4/26/17.
//  Copyright © 2017 Alice Fockele. All rights reserved.
//

#ifndef error_reporting_h
#define error_reporting_h

int get_error_rate(Mat & prediction, Mat & expected);

int get_error_rate(Mat & prediction, Mat & expected){
    //CALCULATE ERROR RATE
    Mat error;
    Mat prediction_int;
    prediction.convertTo(prediction_int, CV_32SC1);
    absdiff(prediction_int, expected, error);
    int num_wrong = countNonZero(error);
    int test_size = expected.rows;
    float error_percent = num_wrong*100 / test_size;
    //   cout << num_wrong << " divide by " << test_size << endl;

    
    
    return error_percent;
}
#endif /* error_reporting_h */
