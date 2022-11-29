//
// Created by Joe Bowser on 2022-11-20.
//

#ifndef ANIMEGANPERFTEST_ANIMEGANPYTORCH_H
#define ANIMEGANPERFTEST_ANIMEGANPYTORCH_H

#include "AnimeGan.h"
#include "torch/script.h"
#include "opencv2/opencv.hpp"
#include "MobileCallGuard.h"
#include <vector>

namespace AnimeGan {


    class AnimeGanPyTorch : AnimeGan {

    public:
        bool loadModel(RunMode runMode, bool isNHWC);
        void doPredict(cv::Mat & input, cv::Mat & output, bool isGPU);
    private:
        mutable torch::jit::script::Module mModule;
        std::string FRAMEWORK;
        std::string PYTORCH_PATH = "/data/data/org.infil00p.animegangallerydemo/files/pytorch/";
        std::string PYTORCH_NCHW_MODEL = "animegan2.pt";
        std::string PYTORCH_VULKAN_NHWC_MODEL = "animegan_vulkan_nhwc.pt";
        bool isNHWC;
    };
}




#endif //ANIMEGANPERFTEST_ANIMEGANPYTORCH_H
