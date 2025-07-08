#include "Utils/OptimizerConfig.hpp"

namespace OptimizerConfig {

    std::unique_ptr<Optimizer> makeOptimizer(const OpenCLSetup& ocl_setup, const OptimizerParameters& params) {
        return params.createOptimizer(ocl_setup);
    }

    SGDOptimizerParameters makeSGDParameters(float learning_rate, float weight_decay_rate) {
        return SGDOptimizerParameters(learning_rate, weight_decay_rate);
    }

    AdamOptimizerParameters makeAdamParameters(float learning_rate, float weight_decay_rate,
                                               float beta1, float beta2, float epsilon) {
        return AdamOptimizerParameters(learning_rate, weight_decay_rate, beta1, beta2, epsilon);
    }

    AdamWOptimizerParameters makeAdamWParameters(float learning_rate, float weight_decay_rate,
                                                 float beta1, float beta2, float epsilon) {
        return AdamWOptimizerParameters(learning_rate, weight_decay_rate, beta1, beta2, epsilon);
    }
}