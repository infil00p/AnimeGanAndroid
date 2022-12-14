/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~ Copyright 2022 Adobe
 ~
 ~ Licensed under the Apache License, Version 2.0 (the "License");
 ~ you may not use this file except in compliance with the License.
 ~ You may obtain a copy of the License at
 ~
 ~     http://www.apache.org/licenses/LICENSE-2.0
 ~
 ~ Unless required by applicable law or agreed to in writing, software
 ~ distributed under the License is distributed on an "AS IS" BASIS,
 ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ~ See the License for the specific language governing permissions and
 ~ limitations under the License.
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "AnimeGanPyTorch.h"


extern "C"
JNIEXPORT jstring JNICALL
Java_org_infil00p_animegangallerydemo_MainActivity_startPredict(JNIEnv *env, jobject thiz,
                                                                jobject buffer,
                                                                jstring external_file_path,
                                                                jint height, jint width) {

    std::string externalPath = std::string(env->GetStringUTFChars(external_file_path, nullptr));
    bool isNHWC=true;
    bool isGPU=false;

    auto now = std::chrono::system_clock::now();
    auto UTC = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    // Endianness matters, ARGB8888 in Android Speak is BGRA in reality
    jbyte* buff = (jbyte*)env->GetDirectBufferAddress(buffer);
    cv::Mat rgbaMat(height, width, CV_8UC4, buff);
    cv::Mat rgbMat, outMat;
    cv::cvtColor(rgbaMat, rgbMat, cv::COLOR_BGRA2RGB);

    std::string input_path = externalPath + "/input_image" + std::to_string(UTC) + ".jpg";
    cv::imwrite(input_path, rgbMat);

    AnimeGan::AnimeGanPyTorch model;
    // We always use NHWC
    model.loadModel(AnimeGan::RunMode::CPU, isNHWC);
    model.doPredict(rgbMat, outMat, isGPU);

    //Figure out where to save the mat
    std::string outputPath = externalPath + "/anime_gan_" + std::to_string(UTC) + ".jpg";
    cv::imwrite(outputPath, outMat);

    //Write to the byte buffer
    return env->NewStringUTF(outputPath.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_org_infil00p_animegangallerydemo_MainActivity_startPredictWithGPU(JNIEnv *env, jobject thiz,
                                                                       jobject buffer,
                                                                       jstring external_file_path,
                                                                       jint height, jint width) {

    std::string externalPath = std::string(env->GetStringUTFChars(external_file_path, nullptr));
    bool isNHWC=true;
    bool isGPU=true;

    auto now = std::chrono::system_clock::now();
    auto UTC = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    // Endianness matters, ARGB8888 in Android Speak is BGRA in reality
    jbyte* buff = (jbyte*)env->GetDirectBufferAddress(buffer);
    cv::Mat rgbaMat(height, width, CV_8UC4, buff);
    cv::Mat rgbMat, outMat;
    cv::cvtColor(rgbaMat, rgbMat, cv::COLOR_BGRA2RGB);

    AnimeGan::AnimeGanPyTorch model;
    // We always use NHWC
    model.loadModel(AnimeGan::RunMode::GPU, isNHWC);
    model.doPredict(rgbMat, outMat, isGPU);

    //Figure out where to save the mat
    std::string outputPath = externalPath + "/anime_gan_" + std::to_string(UTC) + ".png";
    cv::imwrite(outputPath, outMat);

    //Write to the byte buffer
    return env->NewStringUTF(externalPath.c_str());
}