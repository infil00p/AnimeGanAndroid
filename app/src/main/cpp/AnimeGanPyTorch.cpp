//
// Created by Joe Bowser on 2022-11-20.
//

#include "AnimeGanPyTorch.h"
#include "MobileCallGuard.h"

bool AnimeGan::AnimeGanPyTorch::loadModel(RunMode runMode, bool nhwc) {
    auto qengines = at::globalContext().supportedQEngines();
    if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
        qengines.end())
    {
        at::globalContext().setQEngine(at::QEngine::QNNPACK);
    }
    isNHWC = nhwc;

    MobileCallGuard guard;
    // Do the load here, we support CPU and GPU
    mRunMode = runMode;
    if(runMode == RunMode::CPU) {

            mModule = torch::jit::load(PYTORCH_PATH + PYTORCH_NHWC_MODEL);
    }
    else
    {
            mModule = torch::jit::load(PYTORCH_PATH + PYTORCH_VULKAN_NHWC_MODEL);
    }

    mModule.eval();
    return true;
}

void AnimeGan::AnimeGanPyTorch::doPredict(cv::Mat &matInput, cv::Mat &outMat, bool isGPU) {

    cv::Mat preProcesedMat = preProcess(matInput);
    int64_t width = preProcesedMat.cols;
    int64_t height = preProcesedMat.rows;
    float * blob = (float *)(preProcesedMat.data);

    const auto sizes = std::vector<int64_t>{1, 3, 512, 512};
    auto stride_arr = c10::get_channels_last_strides_2d(sizes);
    at::Tensor input;
    try
    {
        input = torch::from_blob(
                blob,
                torch::IntArrayRef(sizes),
                at::TensorOptions(at::kFloat));

        input = input * 2 - 1;
    }
    catch( c10::ValueError e)
    {

    }
    std::vector<torch::jit::IValue> pytorchInputs;

    bool isVulkan = at::is_vulkan_available() && isGPU;
    if(isVulkan)
    {
        auto gpuInputTensor = input.vulkan();
        pytorchInputs.push_back(gpuInputTensor);
    }
    else
    {
        pytorchInputs.push_back(input);
    }

    auto output = [&]() {
        MobileCallGuard guard;

        return mModule.forward(pytorchInputs);
    }();

    if (output.tagKind() == "Tensor") {
        auto tmpTensor = output.toTensor();
        at::Tensor outTensor;
        // This was crashing earlier, this is an API change between 1.9 and 1.12.1
        if (isVulkan) {
            outTensor = tmpTensor.cpu();
        } else {
            outTensor = tmpTensor;
        }

        auto sizes = outTensor.sizes();
        if(sizes.size() == 4 && sizes[1] == 3){
            // Convert the data back to Chunky from Planar
            outTensor = outTensor.permute({0, 3, 1, 2});
        }

        outTensor = outTensor.squeeze().clip(-1, 1) * 0.5 + 0.5;

        cv::Mat outputFloat(cv::Size(sizes[2],sizes[3]), CV_32F, outTensor.data_ptr());
        outMat = postProcess(outputFloat);
    }

}

