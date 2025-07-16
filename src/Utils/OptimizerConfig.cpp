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

    std::unique_ptr<Optimizer> loadOptimizer(const OpenCLSetup& ocl_setup, const H5::Group& optimizer_group) {
        H5::Attribute type_attr = optimizer_group.openAttribute("optimizer_type");
        unsigned int optimizer_type;
        type_attr.read(H5::PredType::NATIVE_UINT, &optimizer_type);

        switch (optimizerTypeFromUint(optimizer_type)) {
            case OptimizerType::SGD:
                return std::make_unique<SGDOptimizer>(ocl_setup, optimizer_group);
            case OptimizerType::Adam:
                return std::make_unique<AdamOptimizer>(ocl_setup, optimizer_group);
            case OptimizerType::AdamW:
                return std::make_unique<AdamWOptimizer>(ocl_setup, optimizer_group);
            default:
                throw std::runtime_error("Unsupported optimizer type in HDF5 file.");
        }
    }
}