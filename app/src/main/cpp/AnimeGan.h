//
// Created by Joe Bowser on 2022-11-20.
//

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
