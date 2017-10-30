# MNIST
Machine learning application to classify handwritten digits, comparison of the support vector machine and multilayer perceptron models

Run main.cpp executes either svm or mlp using either HoG (histogram of gradients) or pixel color count features as inputs. 
You'll first be asked which feature set to use: type either "pixel" or "hog".
Second you'll be asked which model to use: type either mlp or svm.

Best results: 
SVM + HoG: 2% error
MLP + HoG: 3% error

The program will also generate an image of the test set with incorrectly classified digits circled.
The error will also be shown next to the circled digit.
