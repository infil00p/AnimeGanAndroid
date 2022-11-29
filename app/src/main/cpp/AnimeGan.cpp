//
// Created by Joe Bowser on 2022-11-20.
//

#include "AnimeGan.h"
#include <cmath>

AnimeGan::AnimeGan::AnimeGan() {

}

cv::Mat AnimeGan::AnimeGan::preProcess(cv::Mat input) {

    cv::Mat cropped, resizedImage, preMat, out;
    // Resize to something acceptable for the NN to process
    cropped = centerCrop(input);

    cv::resize(cropped,
               resizedImage,
               cv::Size(512, 512),
               cv::InterpolationFlags::INTER_CUBIC);

    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255);
    cv::dnn::blobFromImage(resizedImage, out);

    return out;
}

cv::Mat AnimeGan::AnimeGan::postProcess(cv::Mat input) {
    cv::Mat out;
    // Undo the pre-processing on the inpu

    cv::Mat floatOut = input*255;
    floatOut.convertTo(out, CV_8UC3);
    return out;
}

cv::Mat AnimeGan::AnimeGan::centerCrop(cv::Mat input) {
    int width = input.cols;
    int height = input.rows;

    auto rect = getCenterBox(width, height);

    cv::Mat croppedImage = input(rect);

    return croppedImage;
}

cv::Rect AnimeGan::AnimeGan::getCenterBox(int width, int height) {
    int width_center;
    int width_left = 0;
    int width_right = width;
    int height_center;
    int height_left = 0;
    int height_right = height;

    if(width > height)
    {
        width_center = width/2;
        width_left = width_center - height/2;
        width_right = width_center + height/2;
    }
    else if(height > width)
    {
        height_center = height/2;
        height_left = height_center - height/2;
        height_right = height_center + height/2;
    }

    return cv::Rect(width_left, height_left, width_right, height_right);
}




