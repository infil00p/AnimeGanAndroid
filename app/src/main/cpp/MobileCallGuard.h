//
// Created by Joe Bowser on 2022-11-04.
//

#ifndef ANIMEGAN_MOBILECALLGUARD_H
#define ANIMEGAN_MOBILECALLGUARD_H


#include "torch/script.h"


namespace AnimeGan {

    struct MobileCallGuard
    {
        // AutoGrad is disabled for mobile by default.
        torch::autograd::AutoGradMode no_autograd_guard{false};
        // This needs to be on (taken from the test application)
        torch::AutoNonVariableTypeMode non_var_guard{true};
        // Disable graph optimizer to ensure list of unused ops are not changed for
        // custom mobile build.
        torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
    };
}




#endif //ANIMEGAN_MOBILECALLGUARD_H
