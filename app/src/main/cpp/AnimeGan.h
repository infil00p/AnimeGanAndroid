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

#ifndef ANIMEGANPERFTEST_ANIMEGAN_H
#define ANIMEGANPERFTEST_ANIMEGAN_H

#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

namespace AnimeGan {

    enum RunMode {
        CPU, GPU, NNAPI
    };

    struct TestRunStatus {
        bool completed;
        std::string description;
    };

    // This needs to contain the
    struct ResultSet {
        double duration;
        std::string framework;
        std::string mRunMode;
        std::string isQuant = "false";
    };

class AnimeGan {

public:
    AnimeGan();
    cv::Mat preProcess(cv::Mat input);
    cv::Mat postProcess(cv::Mat input);
    //cv::Mat preProcessFromJFloat(jfloatarray * texArray, bool channelsFirst = true);


protected:
    std::vector<std::vector<unsigned char> > randomData;
    std::vector<ResultSet> lastResults;
    // Not sure what to do with this
    size_t INPUT_SIZE = 512 * 512 * 3 * sizeof(float);
    RunMode mRunMode = CPU;

private:
    cv::Mat centerCrop(cv::Mat input);
    cv::Rect getCenterBox(int width, int height);
};
}




#endif //ANIMEGANPERFTEST_ANIMEGAN_H
