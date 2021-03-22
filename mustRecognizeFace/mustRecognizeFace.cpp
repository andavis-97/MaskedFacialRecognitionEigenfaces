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
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// CIS663 Syracuse University
// Code has been modified by Alison Davis & Ricardo Alonzo Ugalde
// Specializations are commented into the code that make improvements
// to the performance for recognition in COVID-19 facial PPE
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Flag implemented as part of Experiment C
// global for simplicity but ideally would be local
// and image-dependent. Goal would be to one day 
// pass images through another recognizer like 
// Amazon Rekognition that could identify this boolean 
// on a per-image basis and then pass the image to the 
// correct model (croppedModel or normal model depending
// on true/false)
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bool testImagesWithMasks = true;
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, vector<Mat>& testImages, vector<int>& test_labels, char separator = ';') {
    string fn = "C:/Users/aliso/Documents/CIS663/new_small_not_masked.csv";
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
            images.push_back(imread(path, 0));
            std::cout << "Path: " << path << std::endl;
            // This version was tried at a time when the matrices were non-continuous
            //Mat m = imread(path, 0);
            //if (m.empty())
            //{
            //    std::cout << "Empty." << std::endl;
            //}
            //if (m.data == NULL)
            //{
            //    std::cout << "null image" << std::endl;
            //}
            //Mat m2;
            //if (!m.isContinuous())
            //{
            //    std::cout << "Running clone for non-continuous" << std::endl;
            //    m2 = m.clone();
            //}
            //else
            //{
            //    m2 = m;
            //}
            //images.push_back(m2);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }


    string fn_test_samples = "C:/Users/aliso/Documents/CIS663/new_small_masked.csv";
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
            testImages.push_back(imread(path, 0));
            std::cout << "Test Path: " << path << std::endl;
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
    vector<Mat> croppedImages;
    vector<Mat> testImages;
    vector<Mat> croppedTestImages;
    vector<int> labels;
    vector<int> test_labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels, testImages, test_labels);
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
    int height = images[0].rows; //this will get overwritten in the event that MASK-prep eigenfaces runs
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // This is no longer the implementation that is used.
    // This code is kept here as an example but is no longer
    // valuable as many test samples are now used
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::BasicFaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    //Mat testSample = images[images.size() - 1];
    //std::cout << testSample << std::endl;
    //int testLabel = labels[labels.size() - 1];
    //images.pop_back();
    //labels.pop_back();
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // Threshold is implemented in the evaluation step by applying
    // a threshold filter onto the confidence value. This allows the
    // same result set to be analyzed at a variety of thresholds
    // quickly without the time spent re-training the algorithm
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // If you want to use _all_ Eigenfaces and have a threshold,
    // then call the method like this:
    //
    //      EigenFaceRecognizer::create(0, 123.0);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    Ptr<EigenFaceRecognizer> croppedModel = EigenFaceRecognizer::create(); // added capability to create a second cropped model

    try
    {

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            //Modifications from original eigenfaces for Experiment C
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (testImagesWithMasks)
        {
            int l;
            int count = 0;
            for (cv::Mat img : images)
            {
                l = labels[count];
                int height = img.rows;
                int width = img.cols;
                //std::cout << "Height: " << height << std::endl;
                //std::cout << "Width: " << width << std::endl;
                cv::Rect myroi(45, 0, 136, 136);
                cv::Mat croppedImg = img(myroi);
                //std::cout << "Cropped width: " << croppedImg.cols;
                //std::cout << "Cropped height: " << croppedImg.rows;
                //can uncomment this to view the images being resized (needed for sanity check)
                //namedWindow(to_string(l), cv::WINDOW_AUTOSIZE); 
                //imshow(to_string(l), croppedImg);

                if (croppedImg.empty())
                {
                    std::cout << "Empty crop image." << std::endl;
                }
                if (croppedImg.data == NULL)
                {
                    std::cout << "null image" << std::endl;
                }
                Mat m2Crop;
                if (!croppedImg.isContinuous())
                {
                    //Curious if this was necessary, it is.
                    //std::cout << "Making matrix continuous" << std::endl;
                    m2Crop = croppedImg.clone();
                }
                else
                {
                    m2Crop = croppedImg;
                }
                croppedImages.push_back(m2Crop);


                //croppedImages.push_back(croppedImg);
                count++;
            }
            //imshow("cropped image", croppedImages[0]);
            height = croppedImages[0].rows; // reset the height or reshape will fail
            count = 0;
            for (cv::Mat img : testImages)
            {
                l = test_labels[count];
                int height = img.rows;
                int width = img.cols;
                //std::cout << "Height: " << height << std::endl;
                //std::cout << "Width: " << width << std::endl;
                cv::Rect myroi(45, 0, 136, 136);
                cv::Mat croppedImg = img(myroi);
                //std::cout << "Cropped test width: " << croppedImg.cols;
                //std::cout << "Cropped height: " << croppedImg.rows;
                
                // uncomment to show cropped images
                //imshow(to_string(l) + "_test", croppedImg);
                //croppedTestImages.push_back(croppedImg);

                if (croppedImg.empty())
                {
                    std::cout << "Empty crop image." << std::endl;
                }
                if (croppedImg.data == NULL)
                {
                    std::cout << "null image" << std::endl;
                }
                Mat m2Crop;
                if (!croppedImg.isContinuous())
                {
                    //std::cout << "Running clone for non-continuous" << std::endl;
                    m2Crop = croppedImg.clone();
                }
                else
                {
                    m2Crop = croppedImg;
                }
                croppedTestImages.push_back(m2Crop);
                count++;
            }
        }
        //imshow("cropped test image", croppedTestImages[0]);


        //std::cout << "Croppedimages.size: " << croppedImages.size() << std::endl;
        //std::cout << "Croppedtestimages.size: " << croppedTestImages.size() << std::endl;

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (testImagesWithMasks)
        {
            croppedModel->train(croppedImages, labels); //Uncomment this line for MASK-Prep-Eigenfaces
        }
        model->train(images, labels); //this line for traditional Eigenfaces


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
    // testImages and test_labels to generate a larger set of data

        string outfileName = "C:/Users/aliso/Documents/CIS663/results/x.txt"; //Edit this line to the name of your results file
        std::ofstream ofile(outfileName.c_str(), ofstream::out);

        int currentPredictedLabel = -1;
        double currentConfidence = 0.0;
        int current = 0;

        if (testImagesWithMasks)
        {
            for (Mat testSample : croppedTestImages) //uncomment this line for MASK-Prep-Eigenfaces
            {
                //currentPredictedLabel = model->predict(testSample); //Uncomment this line to predict without confidence
                croppedModel->predict(testSample, currentPredictedLabel, currentConfidence); //Uncomment this line to predict with confidence

                // Adding a threshold to the predicted distance so that there is a predicted false case 
                // (did not exist in origianal code, ended up moving to evaluation step so it could be changed faster and evaluated at more thresholds)
                //if (currentConfidence > 10000.00)
                //{
                //    currentPredictedLabel = -1;
                //}
                string result_message = format("%d\t%d\t%f", test_labels[current], currentPredictedLabel, currentConfidence);
                cout << result_message << endl;
                ofile << result_message << endl;
                current++;
            }
        }
        else {
            for (Mat testSample : testImages) //uncomment this line for normal eigenfaces
            {
                //currentPredictedLabel = model->predict(testSample); //Uncomment this line to predict without confidence
                model->predict(testSample, currentPredictedLabel, currentConfidence); //Uncomment this line to predict with confidence

                // Adding a threshold to the predicted distance so that there is a predicted false case 
                // (did not exist in origianal code, ended up moving to evaluation step so it could be changed faster and evaluated at more thresholds)
                //if (currentConfidence > 10000.00)
                //{
                //    currentPredictedLabel = -1;
                //}
                string result_message = format("%d\t%d\t%f", test_labels[current], currentPredictedLabel, currentConfidence);
                cout << result_message << endl;
                ofile << result_message << endl;
                current++;
            }
        }

        //string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
        //cout << result_message << endl;
    }
    catch (cv::Exception& e)
    {
        std::cout << "Failure at try block line 168" << std::endl;
        std::cout << e.msg << std::endl; // output exception message
    }

    Mat eigenvalues;
    if (testImagesWithMasks) {
        eigenvalues = croppedModel->getEigenValues();
    }
    else {
        // Here is how to get the eigenvalues of this Eigenfaces model:
        eigenvalues = model->getEigenValues();
    }
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W;
    // Get the sample mean from the training data
    Mat mean;

    if (testImagesWithMasks) {
        // And we can do the same to display the Eigenvectors (read Eigenfaces):
        W = croppedModel->getEigenVectors();
        // Get the sample mean from the training data
        mean = croppedModel->getMean();
    }
    else {
        // And we can do the same to display the Eigenvectors (read Eigenfaces):
        W = model->getEigenVectors();
        // Get the sample mean from the training data
        mean = model->getMean();
    }

    // Display or save:
    if (argc == 2) {
        try {
            if (testImagesWithMasks) {
                imshow("mean", norm_0_255(mean.reshape(1, croppedImages[0].rows)));
            }
            else {
                imshow("mean", norm_0_255(mean.reshape(1, images[0].rows))); //Uncomment for traditional eigenfaces
              }
        }
        catch (cv::Exception& e)
        {
            std::cout << "Failure at try block line 202" << std::endl;
            std::cout << e.msg << std::endl; // output exception message
        }
    }
    else {
            if (testImagesWithMasks) {
                imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, croppedImages[0].rows)));
            }
            else {
                imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
            }
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
        Mat projection;
        Mat reconstruction;
        if (testImagesWithMasks)
        {
            projection = LDA::subspaceProject(evs, mean, croppedImages[0].reshape(1, 1));
            reconstruction = LDA::subspaceReconstruct(evs, mean, projection);
            reconstruction = norm_0_255(reconstruction.reshape(1, croppedImages[0].rows));
        }
        else
        {
            projection = LDA::subspaceProject(evs, mean, images[0].reshape(1, 1));
            reconstruction = LDA::subspaceReconstruct(evs, mean, projection);
            reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        }

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