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
    preProcesedMat = preProcesedMat * 2 - 1;

    float * blob = (float *)(preProcesedMat.data);

    const auto sizes = std::vector<int64_t>{1, 3, 512, 512};
    auto stride_arr = c10::get_channels_last_strides_2d(sizes);
    at::Tensor input;
    try
    {
        input = torch::from_blob(
                blob,
                torch::IntArrayRef(sizes),
                torch::IntArrayRef(stride_arr),
                at::TensorOptions(at::kFloat)
                        .memory_format(at::MemoryFormat::ChannelsLast));
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

        // Let's do the post-processing entirely in PyTorch
        outTensor = outTensor.squeeze().detach();
        outTensor = outTensor.clip(-1, 1) * 0.5 + 0.5;
        outTensor = outTensor.permute({1, 2, 0}).contiguous();
        outTensor = outTensor.mul(255).clamp(0, 255).to(torch::kU8);

        outMat = cv::Mat(cv::Size(512, 512), CV_8UC3, outTensor.data_ptr());
    }

}

