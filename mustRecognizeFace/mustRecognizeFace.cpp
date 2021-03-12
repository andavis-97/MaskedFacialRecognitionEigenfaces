/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace cv::face;
using namespace std;
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch (src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

string removeSpaces(string str)
{
    // Storing the whole string 
    // into string stream 
    stringstream ss(str);
    string temp;

    // Making the string empty 
    str = "";

    // Running loop till end of stream 
    // and getting every word 
    while (getline(ss, temp, ' ')) {
        // Concatenating in the string 
        // to be returned 
        str = str + temp;
    }
    return str;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, vector<Mat>& test_images, vector<int>& test_labels, char separator = ';') {
    string fn = "C:/Users/aliso/Documents/CIS663/new_5000_not_masked.csv";
    std::ifstream file(fn.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    int linecount = 0;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            //path = removeSpaces(path);
            //std::cout << "path: " << path << std::endl;
            //images.push_back(imread(path, 0));
            std::cout << "Path: " << path << std::endl;
            Mat m = imread(path, 0);
            if (m.empty())
            {
                std::cout << "Empty." << std::endl;
            }
            if (m.data == NULL)
            {
                std::cout << "null image" << std::endl;
            }
            Mat m2;
            if (!m.isContinuous())
            {
                std::cout << "Running close for non-continuous" << std::endl;
                m2 = m.clone();
            }
            else
            {
                m2 = m;
            }
            //cvtColor(m, m2, cv::COLOR_BGR2GRAY);
            images.push_back(m2);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }


    string fn_test_samples = "C:/Users/aliso/Documents/CIS663/new_5000_masked.csv";
    std::ifstream file_test(fn_test_samples.c_str(), ifstream::in);
    if (!file_test) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    //string line, path, classlabel;
    int testlinecount = 0;
    while (getline(file_test, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            //path = removeSpaces(path);
            //std::cout << "path: " << path << std::endl;
            test_images.push_back(imread(path, 0));
            std::cout << "Test Path: " << path << std::endl;
            //Mat mTest = imread(path, 0);
            //if (mTest.empty())
            //{
            //    std::cout << "Empty." << std::endl;
            //}
            //if (mTest.data == NULL)
            //{
            //    std::cout << "null image" << std::endl;
            //}
            //Mat mTest2;
            //if (!mTest.isContinuous())
            //{
            //    std::cout << "Running close for non-continuous" << std::endl;
            //    mTest2 = mTest.clone();
            //}
            //else
            //{
            //    mTest2 = mTest;
            //}
            //cvtColor(m, m2, cv::COLOR_BGR2GRAY);
            //test_images.push_back(mTest2);
            test_labels.push_back(atoi(classlabel.c_str()));
        }
    }
}
int main(int argc, const char* argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
        exit(1);
    }
    string output_folder = ".";
    if (argc == 3) {
        output_folder = string(argv[2]);
    }
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<Mat> test_images;
    vector<int> labels;
    vector<int> test_labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels, test_images, test_labels);
    }
    catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if (images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::BasicFaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[images.size() - 1];
    //std::cout << testSample << std::endl;
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    // The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // This here is a full PCA, if you just want to keep
    // 10 principal components (read Eigenfaces), then call
    // the factory method like this:
    //
    //      EigenFaceRecognizer::create(10);
    //
    // If you want to create a FaceRecognizer with a
    // confidence threshold (e.g. 123.0), call it with:
    //
    //      EigenFaceRecognizer::create(10, 123.0);
    //
    // If you want to use _all_ Eigenfaces and have a threshold,
    // then call the method like this:
    //
    //      EigenFaceRecognizer::create(0, 123.0);
    //
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    try
    {
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:
        //int predictedLabel = model->predict(testSample);
        //
// To get the confidence of a prediction call the model with:
//
//      int predictedLabel = -1;
//      double confidence = 0.0;
//      model->predict(testSample, predictedLabel, confidence);
//

// No longer want just one test sample, want to iterate through 
// test_images and test_labels to generate a larger set of data

    string outfileName = "C:/Users/aliso/Documents/CIS663/results/5000unmaskedTrainMaskedRecog.txt";
    std::ofstream ofile(outfileName.c_str(), ofstream::out);

        int currentPredictedLabel;
        int current = 0;
        for (Mat testSample : test_images)
        {
            currentPredictedLabel = model->predict(testSample);
            string result_message = format("%d\t%d", test_labels[current], currentPredictedLabel);
            cout << result_message << endl;
            ofile << result_message << endl;
            current++;
        }

        //string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
        //cout << result_message << endl;
    }
    catch (cv::Exception& e)
    {
        std::cout << "Failure at try block line 168" << std::endl;
        std::cout << e.msg << std::endl; // output exception message
    }
    // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getEigenVectors();
    // Get the sample mean from the training data
    Mat mean = model->getMean();
    // Display or save:
    if (argc == 2) {
        try {
            imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
        }
        catch (cv::Exception& e)
        {
            std::cout << "Failure at try block line 202" << std::endl;
            std::cout << e.msg << std::endl; // output exception message
        }
    }
    else {
        imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    }
    // Display or save the Eigenfaces:
    for (int i = 0; i < min(10, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        // Display or save:
        if (argc == 2) {
            imshow(format("eigenface_%d", i), cgrayscale);
        }
        else {
            imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        }
    }
    // Display or save the image reconstruction at some predefined steps:
    for (int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components += 15) {
        // slice the eigenvectors from the model
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        Mat projection = LDA::subspaceProject(evs, mean, images[0].reshape(1, 1));
        Mat reconstruction = LDA::subspaceReconstruct(evs, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
        if (argc == 2) {
            imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
        }
        else {
            imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
        }
    }
    // Display if we are not writing to an output folder:
    if (argc == 2) {
        waitKey(0);
    }
    return 0;
}