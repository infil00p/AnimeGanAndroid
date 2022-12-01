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
        std::string PYTORCH_NHWC_MODEL = "animegan2_nhwc.pt";
        std::string PYTORCH_VULKAN_NHWC_MODEL = "animegan_vulkan_nhwc.pt";
        std::string PYTORCH_VULKAN_NCHW_MODEL = "animegan_vulkan_nchw.pt";

        bool isNHWC;
    };
}




#endif //ANIMEGANPERFTEST_ANIMEGANPYTORCH_H
